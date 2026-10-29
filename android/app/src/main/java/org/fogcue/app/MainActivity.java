package org.fogcue.app;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.text.InputType;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public class MainActivity extends Activity {
    private TextView statusView;
    private EditText minutesView;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final Runnable refresh = new Runnable() {
        @Override
        public void run() {
            statusView.setText(LiveService.status + "\n\n" + SensorCheckService.status);
            ui.postDelayed(this, 1000);
        }
    };

    private Button button(LinearLayout box, String label, Runnable r) {
        Button b = new Button(this);
        b.setText(label);
        b.setOnClickListener(v -> r.run());
        box.addView(b);
        return b;
    }

    private void start(Class<?> k, Intent i) {
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(i); else startService(i);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        box.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("FogCue");
        title.setTextSize(20);
        box.addView(title);

        TextView ml = new TextView(this);
        ml.setText("Live run length (minutes)");
        box.addView(ml);
        minutesView = new EditText(this);
        minutesView.setInputType(InputType.TYPE_CLASS_NUMBER);
        minutesView.setText("30");
        box.addView(minutesView);

        button(box, "Start live run", () -> {
            int m = 30;
            try {
                m = Math.max(1, Integer.parseInt(minutesView.getText().toString().trim()));
            } catch (NumberFormatException ignored) {
            }
            Intent i = new Intent(this, LiveService.class).putExtra("mode", "live").putExtra("minutes", m);
            start(LiveService.class, i);
        });
        button(box, "Start energy protocol (2 h)", () -> start(LiveService.class, new Intent(this, LiveService.class).putExtra("mode", "energy")));
        button(box, "Start sensor check", () -> start(SensorCheckService.class, new Intent(this, SensorCheckService.class)));
        button(box, "Stop", () -> {
            stopService(new Intent(this, LiveService.class));
            stopService(new Intent(this, SensorCheckService.class));
        });
        button(box, "Allow running in background", this::requestBatteryExemption);

        statusView = new TextView(this);
        statusView.setTextSize(14);
        statusView.setTextIsSelectable(true);
        statusView.setPadding(0, pad, 0, 0);
        box.addView(statusView);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(box);
        setContentView(scroll);

        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 1);
        }
    }

    private void requestBatteryExemption() {
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        if (pm.isIgnoringBatteryOptimizations(getPackageName())) {
            LiveService.status = "Battery optimisation is already off for this app.";
            return;
        }
        try {
            Intent i = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
            i.setData(Uri.parse("package:" + getPackageName()));
            startActivity(i);
        } catch (Exception e) {
            startActivity(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS));
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        ui.post(refresh);
    }

    @Override
    protected void onPause() {
        super.onPause();
        ui.removeCallbacks(refresh);
    }
}
