package org.fogcue.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Locale;

public class SensorCheckService extends Service implements SensorEventListener {
    static final int[] PERIODS_US = {15625, 10000, 5000, 0};
    static final long STAGE_MS = 60_000L;
    static final long WARMUP_NS = 5_000_000_000L;
    static final String CHANNEL = "fogcue";

    static volatile String status = "Ready. Press \"Start sensor check\".";

    private SensorManager sm;
    private Sensor acc;
    private PowerManager pm;
    private PowerManager.WakeLock wakeLock;
    private HandlerThread thread;
    private Handler handler;
    private File dir;
    private BufferedWriter raw;
    private BufferedWriter summary;
    private int stage = -1;
    private long firstTs = -1;
    private long stageStartWall;
    private int interactiveSec, totalSec;
    private LongList ts = new LongList(), lat = new LongList();
    private boolean active = false;
    private final StringBuilder report = new StringBuilder();

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (active) return START_NOT_STICKY;
        active = true;
        startInForeground();
        pm = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "fogcue:check");
        wakeLock.acquire(PERIODS_US.length * STAGE_MS + 120_000L);
        sm = (SensorManager) getSystemService(SENSOR_SERVICE);
        acc = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        thread = new HandlerThread("fogcue-sensor", android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY);
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
        Notification n = b.setContentTitle("FogCue").setContentText("Sensor check running")
                .setSmallIcon(android.R.drawable.ic_menu_compass).setOngoing(true).build();
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH);
        } else {
            startForeground(1, n);
        }
    }

    private void begin() {
        try {
            if (acc == null) {
                finish("No accelerometer found.");
                return;
            }
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            dir = new File(getExternalFilesDir(null), "sensorcheck_" + stamp);
            if (!dir.mkdirs() && !dir.isDirectory()) {
                finish("Cannot create output folder.");
                return;
            }
            try (BufferedWriter w = new BufferedWriter(new FileWriter(new File(dir, "sensor_info.txt")))) {
                w.write("manufacturer=" + Build.MANUFACTURER + "\nmodel=" + Build.MODEL + "\nsdk=" + Build.VERSION.SDK_INT
                        + "\nrelease=" + Build.VERSION.RELEASE + "\nsensor_name=" + acc.getName() + "\nvendor=" + acc.getVendor()
                        + "\nversion=" + acc.getVersion() + "\nmin_delay_us=" + acc.getMinDelay() + "\nmax_delay_us=" + acc.getMaxDelay()
                        + "\nfifo_reserved=" + acc.getFifoReservedEventCount() + "\nfifo_max=" + acc.getFifoMaxEventCount()
                        + "\nresolution=" + acc.getResolution() + "\nmax_range=" + acc.getMaximumRange() + "\npower_mA=" + acc.getPower()
                        + "\nwake_up=" + acc.isWakeUpSensor() + "\nbattery_exempt=" + pm.isIgnoringBatteryOptimizations(getPackageName()) + "\n");
            }
            summary = new BufferedWriter(new FileWriter(new File(dir, "summary.csv")));
            summary.write("requested_period_us,requested_hz,n_samples,mean_hz,median_dt_ms,sd_dt_ms,min_dt_ms,max_dt_ms,"
                    + "pct_dt_within_5pct,n_gaps_over_1p5x,n_nonpositive_dt,latency_median_ms,latency_p99_ms,latency_max_ms,screen_on_fraction\n");
            report.append("Sensor: ").append(acc.getName()).append(" (").append(acc.getVendor()).append(")\n")
                    .append("Min delay: ").append(acc.getMinDelay()).append(" us\n\n");
            nextStage();
        } catch (IOException e) {
            finish("File error: " + e.getMessage());
        }
    }

    private void nextStage() throws IOException {
        stage++;
        if (stage >= PERIODS_US.length) {
            finish("Done.");
            return;
        }
        ts = new LongList();
        lat = new LongList();
        firstTs = -1;
        interactiveSec = 0;
        totalSec = 0;
        int p = PERIODS_US[stage];
        raw = new BufferedWriter(new FileWriter(new File(dir, "raw_" + (p == 0 ? "fastest" : p + "us") + ".csv")), 1 << 16);
        raw.write("event_ns,arrival_ns,x,y,z\n");
        stageStartWall = SystemClock.elapsedRealtime();
        sm.registerListener(this, acc, p, 0, handler);
        handler.postDelayed(this::tick, 1000);
        handler.postDelayed(this::endStage, STAGE_MS);
    }

    private void tick() {
        if (raw == null) return;
        totalSec++;
        if (pm.isInteractive()) interactiveSec++;
        long left = (STAGE_MS - (SystemClock.elapsedRealtime() - stageStartWall)) / 1000;
        int p = PERIODS_US[stage];
        status = report + "Running stage " + (stage + 1) + "/" + PERIODS_US.length + " (requested "
                + (p == 0 ? "fastest" : String.format(Locale.US, "%.1f Hz", 1e6 / p)) + "), " + Math.max(left, 0) + " s left, "
                + ts.size() + " samples so far.\n\nOutput: " + dir.getAbsolutePath();
        handler.postDelayed(this::tick, 1000);
    }

    @Override
    public void onSensorChanged(SensorEvent e) {
        long arrival = SystemClock.elapsedRealtimeNanos();
        if (raw == null) return;
        try {
            raw.write(e.timestamp + "," + arrival + "," + e.values[0] + "," + e.values[1] + "," + e.values[2] + "\n");
        } catch (IOException ignored) {
        }
        if (firstTs < 0) firstTs = e.timestamp;
        if (e.timestamp - firstTs >= WARMUP_NS) {
            ts.add(e.timestamp);
            lat.add(arrival - e.timestamp);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

    private void endStage() {
        sm.unregisterListener(this);
        handler.removeCallbacksAndMessages(null);
        try {
            raw.close();
            raw = null;
            int p = PERIODS_US[stage];
            String row = summarise(p);
            summary.write(row + "\n");
            summary.flush();
            nextStage();
        } catch (IOException e) {
            finish("File error: " + e.getMessage());
        }
    }

    private String summarise(int p) {
        int n = ts.size();
        String req = p == 0 ? "fastest" : String.format(Locale.US, "%.3f", 1e6 / p);
        double screen = totalSec > 0 ? (double) interactiveSec / totalSec : Double.NaN;
        if (n < 3) {
            report.append("Requested ").append(req).append(" Hz: too few samples (").append(n).append(")\n");
            return p + "," + req + "," + n + ",,,,,,,,,,,," + screen;
        }
        long[] t = ts.toArray();
        double[] dt = new double[n - 1];
        double sum = 0, sum2 = 0;
        int nonpos = 0;
        for (int i = 1; i < n; i++) {
            dt[i - 1] = (t[i] - t[i - 1]) / 1e6;
            sum += dt[i - 1];
            sum2 += dt[i - 1] * dt[i - 1];
            if (t[i] <= t[i - 1]) nonpos++;
        }
        double mean = sum / dt.length;
        double sd = Math.sqrt(Math.max(sum2 / dt.length - mean * mean, 0));
        double[] s = dt.clone();
        Arrays.sort(s);
        double med = percentile(s, 50);
        int within = 0, gaps = 0;
        for (double d : dt) {
            if (Math.abs(d - med) <= 0.05 * med) within++;
            if (d > 1.5 * med) gaps++;
        }
        double hz = (n - 1) / ((t[n - 1] - t[0]) / 1e9);
        long[] l = lat.toArray();
        double[] ls = new double[l.length];
        for (int i = 0; i < l.length; i++) ls[i] = l[i] / 1e6;
        Arrays.sort(ls);
        double pctWithin = 100.0 * within / dt.length;
        report.append(String.format(Locale.US, "Requested %s Hz -> got %.2f Hz, median dt %.3f ms, sd %.3f ms, %.1f%% of intervals within 5%%, %d gaps, latency median %.2f ms\n",
                req, hz, med, sd, pctWithin, gaps, percentile(ls, 50)));
        return String.format(Locale.US, "%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%d,%d,%.6f,%.6f,%.6f,%.3f",
                p, req, n, hz, med, sd, s[0], s[s.length - 1], pctWithin, gaps, nonpos,
                percentile(ls, 50), percentile(ls, 99), ls[ls.length - 1], screen);
    }

    private static double percentile(double[] sorted, double q) {
        if (sorted.length == 0) return Double.NaN;
        double pos = q / 100.0 * (sorted.length - 1);
        int lo = (int) Math.floor(pos), hi = (int) Math.ceil(pos);
        return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
    }

    private void finish(String msg) {
        try {
            if (summary != null) summary.close();
        } catch (IOException ignored) {
        }
        summary = null;
        status = report + "\n" + msg + (dir != null ? "\n\nFiles: " + dir.getAbsolutePath() : "");
        stopForeground(true);
        stopSelf();
    }

    @Override
    public void onDestroy() {
        if (sm != null) sm.unregisterListener(this);
        if (handler != null) {
            handler.removeCallbacksAndMessages(null);
            handler.post(() -> {
                try {
                    if (raw != null) raw.close();
                    if (summary != null) summary.close();
                } catch (IOException ignored) {
                }
                raw = null;
                summary = null;
            });
            thread.quitSafely();
        }
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        if (stage < PERIODS_US.length && stage >= 0 && !status.contains("Done.")) {
            status = report + "\nStopped.";
        }
        active = false;
        super.onDestroy();
    }

    static final class LongList {
        private long[] a = new long[4096];
        private int n = 0;

        void add(long v) {
            if (n == a.length) a = Arrays.copyOf(a, n * 2);
            a[n++] = v;
        }

        int size() {
            return n;
        }

        long[] toArray() {
            return Arrays.copyOf(a, n);
        }
    }
}
