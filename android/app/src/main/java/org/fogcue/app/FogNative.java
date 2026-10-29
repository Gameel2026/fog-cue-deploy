package org.fogcue.app;

public final class FogNative {
    static {
        System.loadLibrary("fogcue");
    }

    private FogNative() {
    }

    public static native long loadModel(byte[] data);

    public static native int[] modelInfo(long model);

    public static native long newController(long model);

    public static native void reset(long controller);

    public static native double step(long controller, double[] window, double[] out);

    public static native long loop(long controller, double[] replay, double seconds);

    public static native void freeController(long controller);

    public static native void freeModel(long model);
}
