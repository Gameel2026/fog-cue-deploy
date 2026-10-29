package org.fogcue.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.ServiceInfo;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class LiveService extends Service implements SensorEventListener {
    static final String CHANNEL = "fogcue";
    static final int SENSOR_PERIOD_US = 10000;
    static final long WARMUP_NS = 10_000_000_000L;
    static final long BLOCK_S = 600;
    static final String[] ENERGY_BLOCKS = {"idle", "live", "idle", "loop", "idle", "live", "idle", "loop", "idle", "live", "idle", "loop"};
    static final int[] MOTION_MINUTES = {5, 15, 25, 35, 45, 55};

    static volatile String status = "";

    private SensorManager sm;
    private Sensor acc;
    private PowerManager pm;
    private PowerManager.WakeLock wakeLock;
    private BatteryManager bm;
    private HandlerThread thread;
    private Handler handler;
    private File dir;
    private BufferedWriter raw, dec, bat, events;
    private long model, controller;
    private int win, fs;
    private double[] replay;
    private String modelHash;
    private StreamProcessor proc;
    private boolean sensorOn = false, active = false;
    private long warmFirst = -1, warmLast = -1, warmCount = 0, decisions = 0, startWall;
    private double maxDelayMs = 0, sumDelayMs = 0;
    private String block = "setup";
    private int blockIndex = -1;
    private volatile boolean looping = false;
    private Thread loopThread;
    private long loopDecisions = 0;
    private String mode;
    private int minutes;
    private ToneGenerator tone;

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (active) return START_NOT_STICKY;
        active = true;
        mode = intent != null && intent.getStringExtra("mode") != null ? intent.getStringExtra("mode") : "live";
        minutes = intent != null ? intent.getIntExtra("minutes", 30) : 30;
        startInForeground();
        pm = (PowerManager) getSystemService(POWER_SERVICE);
        bm = (BatteryManager) getSystemService(BATTERY_SERVICE);
        long totalMs = "energy".equals(mode) ? ENERGY_BLOCKS.length * BLOCK_S * 1000L : minutes * 60_000L;
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "fogcue:live");
        wakeLock.acquire(totalMs + 300_000L);
        sm = (SensorManager) getSystemService(SENSOR_SERVICE);
        acc = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        try {
            tone = new ToneGenerator(AudioManager.STREAM_ALARM, 100);
        } catch (RuntimeException e) {
            tone = null;
        }
        thread = new HandlerThread("fogcue-live", android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY);
        thread.start();
        handler = new Handler(thread.getLooper());
        handler.post(this::begin);
        return START_NOT_STICKY;
    }

    private void startInForeground() {
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(new NotificationChannel(CHANNEL, "FogCue", NotificationManager.IMPORTANCE_LOW));
        }
        Notification.Builder b = Build.VERSION.SDK_INT >= 26 ? new Notification.Builder(this, CHANNEL) : new Notification.Builder(this);
        Notification n = b.setContentTitle("FogCue").setContentText("energy".equals(mode) ? "Energy protocol running" : "Live run running")
                .setSmallIcon(android.R.drawable.ic_menu_compass).setOngoing(true).build();
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(2, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH);
        } else {
            startForeground(2, n);
        }
    }

    private static byte[] readAsset(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] b = new byte[1 << 16];
        int k;
        while ((k = in.read(b)) > 0) out.write(b, 0, k);
        in.close();
        return out.toByteArray();
    }

    private static String sha256(byte[] data) {
        try {
            byte[] h = MessageDigest.getInstance("SHA-256").digest(data);
            StringBuilder s = new StringBuilder();
            for (byte x : h) s.append(String.format(Locale.US, "%02x", x));
            return s.toString();
        } catch (Exception e) {
            return "unavailable";
        }
    }

    private void begin() {
        try {
            if (acc == null) {
                finish("No accelerometer found.");
                return;
            }
            byte[] mb = readAsset(getAssets().open("model.bin"));
            modelHash = sha256(mb);
            model = FogNative.loadModel(mb);
            if (model == 0) {
                finish("Model file could not be loaded.");
                return;
            }
            int[] info = FogNative.modelInfo(model);
            win = info[0];
            fs = info[2];
            controller = FogNative.newController(model);
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            dir = new File(getExternalFilesDir(null), mode + "_" + stamp);
            if (!dir.mkdirs() && !dir.isDirectory()) {
                finish("Cannot create output folder.");
                return;
            }
            raw = writer("raw.csv", "event_ns,arrival_ns,x,y,z,block");
            dec = writer("decisions.csv", "window,last_sample_ns,decision_ns,delay_ms,p,cue_on,processing_ms,block");
            bat = writer("battery.csv", "elapsed_ns,current_ua,current_avg_ua,voltage_mv,level,screen_on,block");
            events = writer("events.csv", "elapsed_ns,event");
            if ("energy".equals(mode)) {
                ByteBuffer bb = ByteBuffer.wrap(readAsset(getAssets().open("replay.bin"))).order(ByteOrder.LITTLE_ENDIAN);
                bb.position(4);
                int n = bb.getInt();
                bb.getInt();
                replay = new double[n * 3];
                for (int i = 0; i < replay.length; i++) replay[i] = bb.getDouble();
            }
            startWall = SystemClock.elapsedRealtime();
            event("start mode=" + mode + " minutes=" + minutes + " model_sha256=" + modelHash);
            handler.post(this::batteryTick);
            if ("energy".equals(mode)) {
                nextBlock();
            } else {
                block = "live";
                startSensor();
                for (int m : MOTION_MINUTES) {
                    if (m + 1 <= minutes) {
                        handler.postDelayed(() -> motion(true), m * 60_000L);
                        handler.postDelayed(() -> motion(false), (m + 1) * 60_000L);
                    }
                }
                handler.postDelayed(() -> finish("Done."), minutes * 60_000L);
            }
        } catch (IOException e) {
            finish("File error: " + e.getMessage());
        }
    }

    private BufferedWriter writer(String name, String header) throws IOException {
        BufferedWriter w = new BufferedWriter(new FileWriter(new File(dir, name)), 1 << 16);
        w.write(header + "\n");
        return w;
    }

    private void event(String s) {
        try {
            if (events != null) {
                events.write(SystemClock.elapsedRealtimeNanos() + "," + s + "\n");
                events.flush();
            }
        } catch (IOException ignored) {
        }
    }

    private void motion(boolean start) {
        event(start ? "motion_start" : "motion_end");
        if (tone != null) tone.startTone(start ? ToneGenerator.TONE_PROP_BEEP2 : ToneGenerator.TONE_PROP_ACK, 600);
    }

    private void startSensor() {
        warmFirst = -1;
        warmCount = 0;
        proc = null;
        FogNative.reset(controller);
        sensorOn = true;
        sm.registerListener(this, acc, SENSOR_PERIOD_US, 0, handler);
        event("sensor_on");
    }

    private void stopSensor() {
        if (!sensorOn) return;
        sm.unregisterListener(this);
        sensorOn = false;
        event("sensor_off");
    }

    private void nextBlock() {
        stopSensor();
        stopLoop();
        blockIndex++;
        if (blockIndex >= ENERGY_BLOCKS.length) {
            finish("Done.");
            return;
        }
        block = ENERGY_BLOCKS[blockIndex];
        event("block_start " + block);
        if ("live".equals(block)) startSensor();
        if ("loop".equals(block)) startLoop();
        handler.postDelayed(this::nextBlock, BLOCK_S * 1000L);
    }

    private void startLoop() {
        looping = true;
        final long ctrl = FogNative.newController(model);
        final long end = SystemClock.elapsedRealtime() + (BLOCK_S - 1) * 1000L;
        loopThread = new Thread(() -> {
            long n = 0;
            while (looping && SystemClock.elapsedRealtime() < end) n += FogNative.loop(ctrl, replay, 1.0);
            synchronized (this) {
                loopDecisions += n;
            }
            event("loop_decisions " + n);
            FogNative.freeController(ctrl);
        }, "fogcue-loop");
        loopThread.start();
    }

    private void stopLoop() {
        looping = false;
        if (loopThread != null) {
            try {
                loopThread.join(10000);
            } catch (InterruptedException ignored) {
            }
            loopThread = null;
        }
    }

    private void batteryTick() {
        if (bat == null) return;
        Intent b = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        int mv = b != null ? b.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) : -1;
        int lvl = b != null ? b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) : -1;
        try {
            bat.write(SystemClock.elapsedRealtimeNanos() + "," + bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) + ","
                    + bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE) + "," + mv + "," + lvl + ","
                    + (pm.isInteractive() ? 1 : 0) + "," + block + "\n");
            bat.flush();
            raw.flush();
            dec.flush();
        } catch (IOException ignored) {
        }
        long el = (SystemClock.elapsedRealtime() - startWall) / 1000;
        status = String.format(Locale.US, "Mode: %s | block: %s | %d s elapsed\nInput: %s\nDecisions: %d | mean delay %.1f ms | max delay %.1f ms\nOutput: %s",
                mode, block, el, proc == null ? "measuring sensor rate" : (proc.isResampling() ? "resampled to 100 Hz" : "native"),
                decisions, decisions > 0 ? sumDelayMs / decisions : 0.0, maxDelayMs, dir.getAbsolutePath());
        handler.postDelayed(this::batteryTick, 1000);
    }

    @Override
    public void onSensorChanged(SensorEvent e) {
        long arrival = SystemClock.elapsedRealtimeNanos();
        if (!sensorOn || raw == null) return;
        try {
            raw.write(e.timestamp + "," + arrival + "," + e.values[0] + "," + e.values[1] + "," + e.values[2] + "," + block + "\n");
        } catch (IOException ignored) {
        }
        if (proc == null) {
            if (warmFirst < 0) warmFirst = e.timestamp;
            warmLast = e.timestamp;
            warmCount++;
            if (warmLast - warmFirst >= WARMUP_NS) {
                double rate = (warmCount - 1) / ((warmLast - warmFirst) / 1e9);
                boolean resample = true;
                event(String.format(Locale.US, "input rate=%.4f path=%s first_sample_after_ns=%d", rate, resample ? "resampled" : "native", e.timestamp));
                proc = new StreamProcessor(controller, win, fs, resample, this::onDecision, SystemClock::elapsedRealtimeNanos);
            }
            return;
        }
        proc.sample(e.timestamp, e.values[0], e.values[1], e.values[2]);
    }

    private void onDecision(long w, long lastNs, long decNs, double p, boolean on, double procMs) {
        double delay = (decNs - lastNs) / 1e6;
        decisions++;
        sumDelayMs += delay;
        if (delay > maxDelayMs) maxDelayMs = delay;
        try {
            dec.write(w + "," + lastNs + "," + decNs + "," + String.format(Locale.US, "%.4f", delay) + ","
                    + String.format(Locale.US, "%.12f", p) + "," + (on ? 1 : 0) + "," + String.format(Locale.US, "%.4f", procMs) + "," + block + "\n");
        } catch (IOException ignored) {
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    private void closeAll() {
        List<BufferedWriter> ws = new ArrayList<>();
        ws.add(raw); ws.add(dec); ws.add(bat); ws.add(events);
        for (BufferedWriter w : ws) {
            try {
                if (w != null) w.close();
            } catch (IOException ignored) {
            }
        }
        raw = dec = bat = events = null;
    }

    private void finish(String msg) {
        stopSensor();
        stopLoop();
        event("finish " + msg + " decisions=" + decisions + " loop_decisions=" + loopDecisions);
        handler.removeCallbacksAndMessages(null);
        closeAll();
        status = msg + "\nDecisions: " + decisions + (dir != null ? "\nFiles: " + dir.getAbsolutePath() : "");
        stopForeground(true);
        stopSelf();
    }

    @Override
    public void onDestroy() {
        if (sm != null) sm.unregisterListener(this);
        sensorOn = false;
        if (handler != null) {
            handler.removeCallbacksAndMessages(null);
            handler.post(() -> {
                if (raw != null) {
                    event("stopped decisions=" + decisions);
                    closeAll();
                    status = "Stopped.\nDecisions: " + decisions + (dir != null ? "\nFiles: " + dir.getAbsolutePath() : "");
                }
                stopLoop();
                if (controller != 0) FogNative.freeController(controller);
                if (model != 0) FogNative.freeModel(model);
                controller = 0;
                model = 0;
            });
            thread.quitSafely();
        }
        if (tone != null) tone.release();
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        active = false;
        super.onDestroy();
    }
}
