#define _GNU_SOURCE
#include "fd_edge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define EPS 1e-12

typedef struct { const unsigned char *p; size_t left; int ok; } Reader;

static void take(Reader *r, void *dst, size_t n) {
    if (!r->ok || r->left < n) { r->ok = 0; return; }
    memcpy(dst, r->p, n); r->p += n; r->left -= n;
}

static void *alloc_take(Reader *r, size_t elem, size_t n) {
    void *p = malloc(elem * (n ? n : 1));
    if (!p) { r->ok = 0; return NULL; }
    take(r, p, elem * n);
    return p;
}

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

FdModel *fd_model_from_memory(const unsigned char *data, size_t size, char *err, size_t errlen) {
    Reader r = {data, size, 1};
    char magic[4]; take(&r, magic, 4);
    if (!r.ok || memcmp(magic, "FOG2", 4)) { snprintf(err, errlen, "bad model magic"); return NULL; }
    FdModel *m = calloc(1, sizeof(FdModel));
    int32_t hdr[6]; take(&r, hdr, sizeof hdr);
    m->nf = hdr[0]; m->H = hdr[1]; m->n_past = hdr[2]; m->win = hdr[3]; m->nfft = hdr[4]; m->nsos = hdr[5];
    double dh[4]; take(&r, dh, sizeof dh);
    m->fs = dh[0]; m->mode_res = dh[1]; m->th_on = dh[2]; m->th_off = dh[3];
    if (!r.ok || m->nf != 24 * FD_NAX || m->H <= 0 || m->H > 256 || m->n_past < 1 || m->n_past > FD_MAX_NPAST
        || m->win < 4 || m->win > 4096 || m->nfft < m->win || m->nsos < 1 || m->nsos > 16) {
        snprintf(err, errlen, "bad model header"); free(m); return NULL;
    }
    int G = 4 * m->H, nf = m->nf, H = m->H;
    m->mu = alloc_take(&r, 8, nf); m->sd = alloc_take(&r, 8, nf);
    m->sos = alloc_take(&r, 8, (size_t)m->nsos * 6); m->zi = alloc_take(&r, 8, (size_t)m->nsos * 2);
    m->Wi0 = alloc_take(&r, 4, (size_t)G * nf); m->Wh0 = alloc_take(&r, 4, (size_t)G * H); m->b0 = alloc_take(&r, 4, G);
    m->Wi1 = alloc_take(&r, 4, (size_t)G * H); m->Wh1 = alloc_take(&r, 4, (size_t)G * H); m->b1 = alloc_take(&r, 4, G);
    m->Wo = alloc_take(&r, 4, (size_t)2 * H); m->bo = alloc_take(&r, 4, 2);
    if (!r.ok || r.left != 0) { snprintf(err, errlen, "model file size mismatch"); fd_model_free(m); return NULL; }
    m->nb = m->nfft / 2 + 1;
    m->hann = malloc(sizeof(double) * m->win); m->hann_ss = 0;
    for (int n = 0; n < m->win; n++) { m->hann[n] = 0.5 - 0.5 * cos(2 * M_PI * n / m->win); m->hann_ss += m->hann[n] * m->hann[n]; }
    m->cosk = malloc(sizeof(double) * m->nb * m->win); m->sink = malloc(sizeof(double) * m->nb * m->win);
    for (int k = 0; k < m->nb; k++)
        for (int n = 0; n < m->win; n++) {
            double a = 2 * M_PI * (double)k * n / m->nfft;
            m->cosk[k * m->win + n] = cos(a); m->sink[k * m->win + n] = -sin(a);
        }
    return m;
}

FdModel *fd_model_from_file(const char *path, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "cannot open model file %s", path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(n > 0 ? n : 1);
    size_t got = fread(buf, 1, n, f); fclose(f);
    FdModel *m = got == (size_t)n ? fd_model_from_memory(buf, n, err, errlen) : NULL;
    if (got != (size_t)n) snprintf(err, errlen, "cannot read model file");
    free(buf); return m;
}

void fd_model_free(FdModel *m) {
    if (!m) return;
    free(m->mu); free(m->sd); free(m->sos); free(m->zi); free(m->Wi0); free(m->Wh0); free(m->b0);
    free(m->Wi1); free(m->Wh1); free(m->b1); free(m->Wo); free(m->bo); free(m->hann); free(m->cosk); free(m->sink);
    free(m);
}

FdCtrl *fd_ctrl_new(const FdModel *m) {
    FdCtrl *c = calloc(1, sizeof(FdCtrl)); c->m = m;
    int H = m->H, G = 4 * H, T = m->n_past, nf = m->nf;
    c->z = calloc((size_t)m->nsos * 2 * FD_NAX, sizeof(double));
    c->buf = calloc((size_t)T * nf, sizeof(float)); c->X = calloc((size_t)T * nf, sizeof(float));
    c->Y1 = calloc((size_t)T * H, sizeof(float)); c->Y2 = calloc((size_t)T * H, sizeof(float));
    c->gates = calloc(G, sizeof(float)); c->h = calloc(H, sizeof(float)); c->c = calloc(H, sizeof(float));
    c->y = calloc((size_t)m->win * FD_NAX, sizeof(double)); c->col = calloc(m->win, sizeof(double));
    c->feat = calloc(nf, sizeof(double)); c->sorted = calloc(m->win, sizeof(double)); c->P = calloc(m->nb, sizeof(double));
    return c;
}

void fd_ctrl_reset(FdCtrl *c) { c->started = 0; c->nbuf = 0; c->on = 0; }

void fd_ctrl_free(FdCtrl *c) {
    if (!c) return;
    free(c->z); free(c->buf); free(c->X); free(c->Y1); free(c->Y2); free(c->gates); free(c->h); free(c->c);
    free(c->y); free(c->col); free(c->feat); free(c->sorted); free(c->P); free(c);
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void stat_feats(FdCtrl *c, const double *x, double *o) {
    const FdModel *m = c->m; int W = m->win; double *s = c->sorted;
    double mean = 0, m2 = 0, m3 = 0, m4 = 0, ha = 0, sq = 0, mad = 0, medad = 0;
    for (int i = 0; i < W; i++) mean += x[i];
    mean /= W;
    for (int i = 0; i < W; i++) {
        double d = x[i] - mean; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d;
        ha += 1.0 / (fabs(x[i]) + EPS); sq += x[i] * x[i]; mad += fabs(d);
    }
    memcpy(s, x, sizeof(double) * W); qsort(s, W, sizeof(double), cmpd);
    double med = (W % 2) ? s[W / 2] : 0.5 * (s[W / 2 - 1] + s[W / 2]);
    double mn = s[0], mx = s[W - 1];
    for (int i = 0; i < W; i++) medad += fabs(x[i] - med);
    for (int i = 0; i < W; i++) s[i] = rint(x[i] / m->mode_res);
    qsort(s, W, sizeof(double), cmpd);
    double best = s[0]; int bc = 0;
    for (int i = 0; i < W;) { int j = i; while (j < W && s[j] == s[i]) j++; if (j - i > bc) { bc = j - i; best = s[i]; } i = j; }
    double m2p = m2 / W, var1 = m2 / (W - 1);
    o[0] = mean; o[1] = med; o[2] = best * m->mode_res; o[3] = mn; o[4] = mx; o[5] = mx - mn;
    o[6] = W / ha; o[7] = sqrt(var1); o[8] = var1; o[9] = mad / W; o[10] = medad / W;
    o[11] = m2p > 0 ? (m4 / W) / (m2p * m2p) : NAN;
    o[12] = m2p > 0 ? (m3 / W) / pow(m2p, 1.5) : NAN;
    o[13] = sqrt(sq / W);
}

static void spec_feats(FdCtrl *c, const double *x, double *o) {
    const FdModel *m = c->m; int W = m->win, NB = m->nb; double *P = c->P;
    double mean = 0;
    for (int i = 0; i < W; i++) mean += x[i];
    mean /= W;
    double norm = m->fs * m->hann_ss, tot = 0, df = m->fs / m->nfft;
    for (int k = 0; k < NB; k++) {
        double re = 0, im = 0; const double *ck = m->cosk + (size_t)k * W, *sk = m->sink + (size_t)k * W;
        for (int n = 0; n < W; n++) { double v = (x[n] - mean) * m->hann[n]; re += v * ck[n]; im += v * sk[n]; }
        P[k] = (re * re + im * im) / norm;
        if (k > 0 && k < NB - 1) P[k] *= 2;
        tot += P[k];
    }
    double loco = 0, frz = 0, mean_f = 0, msum = 0, cen = 0, var = 0, kurt = 0, ent = 0, cs = 0;
    for (int k = 0; k < NB; k++) {
        double f = k * df;
        if (f >= 0.5 && f < 3.0) loco += P[k];
        if (f >= 3.0 && f < 8.0) frz += P[k];
        msum += sqrt(P[k]);
    }
    loco *= df; frz *= df;
    double den = tot + EPS, mden = msum + EPS;
    for (int k = 0; k < NB; k++) { double p = P[k] / den; mean_f += k * df * p; cen += k * df * (sqrt(P[k]) / mden); }
    for (int k = 0; k < NB; k++) { double p = P[k] / den, d = k * df - mean_f; var += d * d * p; kurt += d * d * d * d * p; ent += p * log2(p + EPS); }
    double sd = sqrt(var) + EPS;
    int med = NB - 1, ip = 0;
    for (int k = 0; k < NB; k++) { cs += P[k]; if (cs >= 0.5 * tot) { med = k; break; } }
    for (int k = 1; k < NB; k++) if (P[k] > P[ip]) ip = k;
    o[0] = loco; o[1] = frz; o[2] = frz / (loco + EPS); o[3] = frz + loco; o[4] = mean_f; o[5] = med * df;
    o[6] = cen; o[7] = kurt / (sd * sd * sd * sd); o[8] = -ent / log2((double)NB); o[9] = ip * df;
}

void fd_features(FdCtrl *c, const double *y, double *f) {
    const FdModel *m = c->m;
    for (int a = 0; a < FD_NAX; a++) {
        for (int n = 0; n < m->win; n++) c->col[n] = y[n * FD_NAX + a];
        stat_feats(c, c->col, f + a * 24); spec_feats(c, c->col, f + a * 24 + 14);
    }
}

static inline float sigf(float x) { return 1.0f / (1.0f + expf(-x)); }

static void lstm_layer(FdCtrl *c, const float *X, int nin, const float *Wi, const float *Wh, const float *b, float *Y) {
    int H = c->m->H, G = 4 * H, T = c->m->n_past; float *h = c->h, *cc = c->c, *g = c->gates;
    memset(h, 0, sizeof(float) * H); memset(cc, 0, sizeof(float) * H);
    for (int t = 0; t < T; t++) {
        const float *x = X + (size_t)t * nin;
        for (int r = 0; r < G; r++) {
            float s = 0; const float *wi = Wi + (size_t)r * nin; for (int j = 0; j < nin; j++) s += wi[j] * x[j];
            s += b[r];
            float u = 0; const float *wh = Wh + (size_t)r * H; for (int j = 0; j < H; j++) u += wh[j] * h[j];
            g[r] = s + u;
        }
        for (int j = 0; j < H; j++) {
            float i = sigf(g[j]), f = sigf(g[H + j]), gg = tanhf(g[2 * H + j]), o = sigf(g[3 * H + j]);
            cc[j] = f * cc[j] + i * gg; h[j] = o * tanhf(cc[j]); Y[(size_t)t * H + j] = h[j];
        }
    }
}

double fd_ctrl_step(FdCtrl *c, const double *raw, double *st) {
    const FdModel *m = c->m; int W = m->win, nf = m->nf, T = m->n_past, H = m->H, NS = m->nsos;
    double t0 = st ? now_s() : 0;
    if (!c->started) {
        for (int s = 0; s < NS; s++) for (int a = 0; a < FD_NAX; a++) {
            c->z[(s * 2 + 0) * FD_NAX + a] = m->zi[s * 2 + 0] * raw[a];
            c->z[(s * 2 + 1) * FD_NAX + a] = m->zi[s * 2 + 1] * raw[a];
        }
        c->started = 1;
    }
    for (int n = 0; n < W; n++)
        for (int a = 0; a < FD_NAX; a++) {
            double v = raw[n * FD_NAX + a];
            for (int s = 0; s < NS; s++) {
                const double *q = m->sos + s * 6; double *z0 = &c->z[(s * 2 + 0) * FD_NAX + a], *z1 = &c->z[(s * 2 + 1) * FD_NAX + a];
                double out = q[0] * v + *z0;
                *z0 = q[1] * v - q[4] * out + *z1; *z1 = q[2] * v - q[5] * out; v = out;
            }
            c->y[n * FD_NAX + a] = v;
        }
    double t1 = st ? now_s() : 0;
    fd_features(c, c->y, c->feat);
    float *zrow;
    if (c->nbuf == T) { memmove(c->buf, c->buf + nf, sizeof(float) * nf * (T - 1)); c->nbuf--; }
    zrow = c->buf + (size_t)c->nbuf * nf;
    for (int k = 0; k < nf; k++) {
        double v = isfinite(c->feat[k]) ? c->feat[k] : 0.0;
        v = (v - m->mu[k]) / m->sd[k]; if (v > 10) v = 10; if (v < -10) v = -10; zrow[k] = (float)v;
    }
    c->nbuf++;
    double t2 = st ? now_s() : 0;
    memset(c->X, 0, sizeof(float) * T * nf);
    memcpy(c->X + (size_t)(T - c->nbuf) * nf, c->buf, sizeof(float) * nf * c->nbuf);
    lstm_layer(c, c->X, nf, m->Wi0, m->Wh0, m->b0, c->Y1);
    lstm_layer(c, c->Y1, H, m->Wi1, m->Wh1, m->b1, c->Y2);
    float zo[2]; const float *hl = c->Y2 + (size_t)(T - 1) * H;
    for (int r = 0; r < 2; r++) { float s = 0; for (int j = 0; j < H; j++) s += m->Wo[r * H + j] * hl[j]; zo[r] = s + m->bo[r]; }
    float mx = zo[0] > zo[1] ? zo[0] : zo[1]; float e0 = expf(zo[0] - mx), e1 = expf(zo[1] - mx);
    double p = (double)(e1 / (e0 + e1));
    c->on = c->on ? (p >= m->th_off) : (p >= m->th_on);
    if (st) { double t3 = now_s(); st[0] = t1 - t0; st[1] = t2 - t1; st[2] = t3 - t2; }
    return p;
}
