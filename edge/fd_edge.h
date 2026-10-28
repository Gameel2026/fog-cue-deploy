#ifndef FD_EDGE_H
#define FD_EDGE_H
#include <stddef.h>

#define FD_NAX 3
#define FD_MAX_NPAST 32

typedef struct {
    int nf, H, n_past, win, nfft, nsos, nb;
    double fs, mode_res, th_on, th_off;
    double *mu, *sd, *sos, *zi;
    float *Wi0, *Wh0, *b0, *Wi1, *Wh1, *b1, *Wo, *bo;
    double *hann, hann_ss, *cosk, *sink;
    float *WiT0, *WhT0, *WiT1, *WhT1;
} FdModel;

typedef struct {
    const FdModel *m;
    double *z;
    float *buf, *proj, *full, *P2, *Y1, *Y2, *gates, *u, *h, *c;
    double *y, *col, *feat, *sorted, *P;
    int started, nbuf, on;
} FdCtrl;

FdModel *fd_model_from_memory(const unsigned char *data, size_t size, char *err, size_t errlen);
FdModel *fd_model_from_file(const char *path, char *err, size_t errlen);
void fd_model_free(FdModel *m);
FdCtrl *fd_ctrl_new(const FdModel *m);
void fd_ctrl_reset(FdCtrl *c);
void fd_ctrl_free(FdCtrl *c);
double fd_ctrl_step(FdCtrl *c, const double *raw, double *stage_seconds);
void fd_features(FdCtrl *c, const double *filtered, double *out);

#endif
