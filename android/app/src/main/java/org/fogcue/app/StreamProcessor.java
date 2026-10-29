package org.fogcue.app;

public final class StreamProcessor {
    public interface Sink {
        void decision(long window, long lastSampleNs, long decisionNs, double p, boolean on, double processingMs);
    }

    public interface Clock {
        long nowNs();
    }

    public static final double G = 9.80665;

    private final long controller;
    private final int win;
    private final boolean resample;
    private final long periodNs;
    private final double[] buf;
    private final double[] out = new double[2];
    private final Sink sink;
    private final Clock clock;
    private int n = 0;
    private long window = 0;
    private boolean havePrev = false;
    private long prevT, nextGrid;
    private double pv, pm, pa;

    public StreamProcessor(long controller, int win, int fs, boolean resample, Sink sink, Clock clock) {
        this.controller = controller;
        this.win = win;
        this.resample = resample;
        this.periodNs = Math.round(1e9 / fs);
        this.buf = new double[win * 3];
        this.sink = sink;
        this.clock = clock;
    }

    public boolean isResampling() {
        return resample;
    }

    public void sample(long tNs, float x, float y, float z) {
        double v = -y / G, ml = x / G, ap = -z / G;
        if (!resample) {
            push(v, ml, ap, tNs);
            return;
        }
        if (!havePrev) {
            havePrev = true;
            prevT = tNs; pv = v; pm = ml; pa = ap;
            nextGrid = tNs;
            push(v, ml, ap, tNs);
            nextGrid += periodNs;
            return;
        }
        if (tNs <= prevT) return;
        while (nextGrid <= tNs) {
            double w = (double) (nextGrid - prevT) / (double) (tNs - prevT);
            push(pv + w * (v - pv), pm + w * (ml - pm), pa + w * (ap - pa), tNs);
            nextGrid += periodNs;
        }
        prevT = tNs; pv = v; pm = ml; pa = ap;
    }

    private void push(double v, double ml, double ap, long sourceNs) {
        buf[3 * n] = v;
        buf[3 * n + 1] = ml;
        buf[3 * n + 2] = ap;
        n++;
        if (n == win) {
            double p = FogNative.step(controller, buf, out);
            long t = clock.nowNs();
            sink.decision(window++, sourceNs, t, p, out[0] != 0.0, out[1]);
            n = 0;
        }
    }
}
