#include <jni.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "fd_edge.h"

static double mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

JNIEXPORT jlong JNICALL Java_org_fogcue_app_FogNative_loadModel(JNIEnv *env, jclass cls, jbyteArray data) {
    jsize n = (*env)->GetArrayLength(env, data);
    jbyte *b = (*env)->GetByteArrayElements(env, data, NULL);
    char err[256];
    FdModel *m = fd_model_from_memory((const unsigned char *)b, (size_t)n, err, sizeof err);
    (*env)->ReleaseByteArrayElements(env, data, b, JNI_ABORT);
    return (jlong)(intptr_t)m;
}

JNIEXPORT jintArray JNICALL Java_org_fogcue_app_FogNative_modelInfo(JNIEnv *env, jclass cls, jlong mh) {
    FdModel *m = (FdModel *)(intptr_t)mh;
    jint v[4] = {m->win, m->n_past, (jint)(m->fs + 0.5), m->H};
    jintArray a = (*env)->NewIntArray(env, 4);
    (*env)->SetIntArrayRegion(env, a, 0, 4, v);
    return a;
}

JNIEXPORT jlong JNICALL Java_org_fogcue_app_FogNative_newController(JNIEnv *env, jclass cls, jlong mh) {
    return (jlong)(intptr_t)fd_ctrl_new((FdModel *)(intptr_t)mh);
}

JNIEXPORT void JNICALL Java_org_fogcue_app_FogNative_reset(JNIEnv *env, jclass cls, jlong ch) {
    fd_ctrl_reset((FdCtrl *)(intptr_t)ch);
}

JNIEXPORT jdouble JNICALL Java_org_fogcue_app_FogNative_step(JNIEnv *env, jclass cls, jlong ch, jdoubleArray raw, jdoubleArray out) {
    FdCtrl *c = (FdCtrl *)(intptr_t)ch;
    jdouble *r = (*env)->GetDoubleArrayElements(env, raw, NULL);
    double t0 = mono();
    double p = fd_ctrl_step(c, r, NULL);
    double t1 = mono();
    (*env)->ReleaseDoubleArrayElements(env, raw, r, JNI_ABORT);
    jdouble o[2] = {(jdouble)c->on, 1e3 * (t1 - t0)};
    (*env)->SetDoubleArrayRegion(env, out, 0, 2, o);
    return p;
}

JNIEXPORT jlong JNICALL Java_org_fogcue_app_FogNative_loop(JNIEnv *env, jclass cls, jlong ch, jdoubleArray replay, jdouble seconds) {
    FdCtrl *c = (FdCtrl *)(intptr_t)ch;
    jsize n = (*env)->GetArrayLength(env, replay);
    jdouble *r = (*env)->GetDoubleArrayElements(env, replay, NULL);
    int W = c->m->win; long nw = n / (W * FD_NAX), k = 0;
    double t0 = mono();
    while (nw > 0 && mono() - t0 < seconds) {
        if (k % nw == 0) fd_ctrl_reset(c);
        fd_ctrl_step(c, r + (size_t)(k % nw) * W * FD_NAX, NULL); k++;
    }
    (*env)->ReleaseDoubleArrayElements(env, replay, r, JNI_ABORT);
    return (jlong)k;
}

JNIEXPORT void JNICALL Java_org_fogcue_app_FogNative_freeController(JNIEnv *env, jclass cls, jlong ch) {
    fd_ctrl_free((FdCtrl *)(intptr_t)ch);
}

JNIEXPORT void JNICALL Java_org_fogcue_app_FogNative_freeModel(JNIEnv *env, jclass cls, jlong mh) {
    fd_model_free((FdModel *)(intptr_t)mh);
}
