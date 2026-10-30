#define _GNU_SOURCE
#include "fd_edge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define PORT 50007
#define PKT_SAMPLES 4
#define PKT_SZ (4 + PKT_SAMPLES * FD_NAX * 8)
#define REP_SZ 13

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(1); }
static void rd(FILE *f, void *p, size_t sz, size_t n) { if (fread(p, sz, n, f) != n) die("input file too short"); }
static double now_s(clockid_t id) { struct timespec t; clock_gettime(id, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double peak_rss_mb(void) { struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss / 1024.0; }

static FdModel *load(const char *path) {
    char err[256]; FdModel *m = fd_model_from_file(path, err, sizeof err);
    if (!m) die(err);
    return m;
}

typedef struct { int n, np; double *raw, *p_ref; } Replay;

static Replay load_replay(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) die("cannot open replay file");
    char magic[4]; rd(f, magic, 1, 4); if (memcmp(magic, "FOGR", 4)) die("bad replay magic");
    Replay r; int32_t n, np; rd(f, &n, 4, 1); rd(f, &np, 4, 1); r.n = n; r.np = np;
    r.raw = malloc(sizeof(double) * (size_t)n * FD_NAX); rd(f, r.raw, 8, (size_t)n * FD_NAX);
    r.p_ref = malloc(sizeof(double) * (np > 0 ? np : 1)); if (np > 0) rd(f, r.p_ref, 8, np);
    fclose(f); return r;
}

static int mode_bench(FdModel *m, const char *rp, const char *pre, int repeats) {
    Replay r = load_replay(rp); int W = m->win, nw = r.n / W; char fn[512];
    snprintf(fn, sizeof fn, "%s_bench.csv", pre); FILE *o = fopen(fn, "w"); if (!o) die("cannot write output");
    fprintf(o, "pass,window,p,cue_on,p_ref,total_ms,filter_ms,features_ms,lstm_ms\n");
    FdCtrl *c = fd_ctrl_new(m);
    double c0 = (double)clock() / CLOCKS_PER_SEC, w0 = now_s(CLOCK_MONOTONIC);
    for (int pass = 0; pass < repeats; pass++) {
        fd_ctrl_reset(c);
        for (int k = 0; k < nw; k++) {
            double st[3], t0 = now_s(CLOCK_MONOTONIC); double p = fd_ctrl_step(c, r.raw + (size_t)k * W * FD_NAX, st); double t1 = now_s(CLOCK_MONOTONIC);
            fprintf(o, "%d,%d,%.12f,%d,%.12f,%.6f,%.6f,%.6f,%.6f\n", pass, k, p, c->on, k < r.np ? r.p_ref[k] : NAN,
                    1e3 * (t1 - t0), 1e3 * st[0], 1e3 * st[1], 1e3 * st[2]);
        }
    }
    fclose(o);
    printf("bench: %d passes x %d windows; wall %.2f s, cpu %.2f s, peak memory %.1f MB -> %s\n", repeats, nw,
           now_s(CLOCK_MONOTONIC) - w0, (double)clock() / CLOCKS_PER_SEC - c0, peak_rss_mb(), fn);
    fd_ctrl_free(c); return 0;
}

static int mode_loop(FdModel *m, const char *rp, double seconds, const char *pre) {
    Replay r = load_replay(rp); int W = m->win, nw_rec = r.n / W;
    FdCtrl *c = fd_ctrl_new(m); long k = 0;
    double w0 = now_s(CLOCK_MONOTONIC), c0 = (double)clock() / CLOCKS_PER_SEC;
    while (now_s(CLOCK_MONOTONIC) - w0 < seconds) {
        if (k % nw_rec == 0) fd_ctrl_reset(c);
        fd_ctrl_step(c, r.raw + (size_t)(k % nw_rec) * W * FD_NAX, NULL); k++;
    }
    double wall = now_s(CLOCK_MONOTONIC) - w0, cpu = (double)clock() / CLOCKS_PER_SEC - c0;
    char fn[512]; snprintf(fn, sizeof fn, "%s_loop.csv", pre); FILE *o = fopen(fn, "w"); if (!o) die("cannot write output");
    fprintf(o, "decisions,wall_s,cpu_s,decisions_per_s\n%ld,%.6f,%.6f,%.3f\n", k, wall, cpu, k / wall); fclose(o);
    printf("loop: %ld decisions in %.1f s (%.1f per s) -> %s\n", k, wall, k / wall, fn);
    fd_ctrl_free(c); return 0;
}

static int mode_realtime(FdModel *m, const char *rp, double minutes, const char *pre) {
    Replay r = load_replay(rp); int W = m->win, nw_rec = r.n / W;
    long nw = lround(minutes * 60 * m->fs / W); char fn[512];
    double period = W / m->fs;
    snprintf(fn, sizeof fn, "%s_realtime.csv", pre); FILE *o = fopen(fn, "w"); if (!o) die("cannot write output");
    fprintf(o, "window,p,cue_on,processing_ms,delay_monotonic_ms,delay_boottime_ms\n");
    FdCtrl *c = fd_ctrl_new(m);
    double mono0 = now_s(CLOCK_MONOTONIC) + 1.0, boot0 = now_s(CLOCK_BOOTTIME) + 1.0, cpu0 = (double)clock() / CLOCKS_PER_SEC;
    for (long k = 0; k < nw; k++) {
        double target = mono0 + (k + 1) * period; struct timespec ts;
        ts.tv_sec = (time_t)target; ts.tv_nsec = (long)((target - ts.tv_sec) * 1e9);
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
        double a = now_s(CLOCK_MONOTONIC);
        if (k % nw_rec == 0) fd_ctrl_reset(c);
        double p = fd_ctrl_step(c, r.raw + (size_t)(k % nw_rec) * W * FD_NAX, NULL);
        double b = now_s(CLOCK_MONOTONIC), bb = now_s(CLOCK_BOOTTIME);
        fprintf(o, "%ld,%.12f,%d,%.6f,%.6f,%.6f\n", k, p, c->on, 1e3 * (b - a), 1e3 * (b - target), 1e3 * (bb - (boot0 + (k + 1) * period)));
        if (k % 600 == 0) { printf("window %ld/%ld  p=%.3f  cue=%s\n", k, nw, p, c->on ? "ON" : "off"); fflush(stdout); }
    }
    fclose(o);
    double wall = now_s(CLOCK_MONOTONIC) - (mono0 - 1.0), cpu = (double)clock() / CLOCKS_PER_SEC - cpu0;
    printf("realtime: %ld decisions; wall %.1f s, cpu %.2f s (duty cycle %.2f %%), peak memory %.1f MB -> %s\n",
           nw, wall, cpu, 100 * cpu / wall, peak_rss_mb(), fn);
    snprintf(fn, sizeof fn, "%s_realtime_run.csv", pre); o = fopen(fn, "w"); if (!o) die("cannot write output");
    fprintf(o, "wall_s,cpu_s,boottime_s,peak_memory_mb\n%.6f,%.6f,%.6f,%.3f\n", wall, cpu, now_s(CLOCK_BOOTTIME) - (boot0 - 1.0), peak_rss_mb());
    fclose(o); fd_ctrl_free(c); return 0;
}

static int mode_dataset(FdModel *m, const char *sp, const char *out) {
    FILE *f = fopen(sp, "rb"); if (!f) die("cannot open streams file");
    char magic[4]; rd(f, magic, 1, 4); if (memcmp(magic, "FOGS", 4)) die("bad streams magic");
    int32_t ns, win; rd(f, &ns, 4, 1); rd(f, &win, 4, 1);
    if (win != m->win) die("streams window length does not match the model");
    FILE *o = fopen(out, "w"); if (!o) die("cannot write output");
    fprintf(o, "stretch,window,p,cue_on\n");
    FdCtrl *c = fd_ctrl_new(m); long tot = 0;
    for (int s = 0; s < ns; s++) {
        int32_t nw; rd(f, &nw, 4, 1);
        double *raw = malloc(sizeof(double) * (size_t)nw * m->win * FD_NAX); rd(f, raw, 8, (size_t)nw * m->win * FD_NAX);
        fd_ctrl_reset(c);
        for (int k = 0; k < nw; k++) { double p = fd_ctrl_step(c, raw + (size_t)k * m->win * FD_NAX, NULL); fprintf(o, "%d,%d,%.12f,%d\n", s, k, p, c->on); }
        tot += nw; free(raw);
    }
    fclose(f); fclose(o); fd_ctrl_free(c);
    printf("dataset: %d stretches, %ld windows -> %s\n", ns, tot, out); return 0;
}

static int mode_stream(FdModel *m, const char *peer, const char *pre) {
    int s;
    if (strcmp(peer, "listen") == 0) {
        int srv = socket(AF_INET, SOCK_STREAM, 0), one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(PORT); a.sin_addr.s_addr = INADDR_ANY;
        if (bind(srv, (struct sockaddr *)&a, sizeof a) < 0 || listen(srv, 1) < 0) die("cannot listen on port 50007");
        printf("waiting on port %d ... (start the sender on the laptop)\n", PORT); fflush(stdout);
        s = accept(srv, NULL, NULL); close(srv);
    } else {
        struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(PORT);
        if (inet_pton(AF_INET, peer, &a.sin_addr) != 1) die("bad laptop IP address");
        printf("connecting to %s:%d ...\n", peer, PORT); fflush(stdout);
        for (;;) { s = socket(AF_INET, SOCK_STREAM, 0); if (connect(s, (struct sockaddr *)&a, sizeof a) == 0) break; close(s); sleep(1); }
    }
    int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    printf("connected\n"); fflush(stdout);
    char fn[512]; snprintf(fn, sizeof fn, "%s_stream.csv", pre); FILE *o = fopen(fn, "w");
    snprintf(fn, sizeof fn, "%s_stream_samples.bin", pre); FILE *sm = fopen(fn, "wb");
    if (!o || !sm) die("cannot write output");
    fprintf(o, "window,p,cue_on,processing_ms,on_phone_delay_ms,arrival_s,boottime_s,seq_errors_so_far\n");
    static unsigned char buf[1 << 17]; size_t have = 0; int W = m->win;
    double *win = malloc(sizeof(double) * W * FD_NAX); int nin = 0; long k = 0; uint32_t expect = 0; long seqerr = 0;
    FdCtrl *c = fd_ctrl_new(m);
    double w0 = now_s(CLOCK_MONOTONIC), b0 = now_s(CLOCK_BOOTTIME), cpu0 = (double)clock() / CLOCKS_PER_SEC;
    for (;;) {
        ssize_t got = recv(s, buf + have, sizeof buf - have, 0);
        if (got <= 0) break;
        double t_arr = now_s(CLOCK_MONOTONIC); have += (size_t)got; size_t off = 0;
        while (have - off >= PKT_SZ) {
            uint32_t seq; memcpy(&seq, buf + off, 4); if (seq != expect) seqerr++; expect = seq + 1;
            for (int j = 0; j < PKT_SAMPLES; j++) {
                memcpy(win + nin * FD_NAX, buf + off + 4 + j * FD_NAX * 8, FD_NAX * 8); nin++;
                if (nin == W) {
                    fwrite(win, sizeof(double), W * FD_NAX, sm);
                    double t1 = now_s(CLOCK_MONOTONIC); double p = fd_ctrl_step(c, win, NULL); double t2 = now_s(CLOCK_MONOTONIC);
                    unsigned char rep[REP_SZ]; uint32_t kk = (uint32_t)k; memcpy(rep, &kk, 4); rep[4] = (unsigned char)c->on; memcpy(rep + 5, &p, 8);
                    if (send(s, rep, REP_SZ, MSG_NOSIGNAL) != REP_SZ) fprintf(stderr, "send failed at window %ld\n", k);
                    fprintf(o, "%ld,%.12f,%d,%.6f,%.6f,%.6f,%.6f,%ld\n", k, p, c->on, 1e3 * (t2 - t1), 1e3 * (t2 - t_arr), t_arr - w0, now_s(CLOCK_BOOTTIME) - b0, seqerr);
                    if (k % 600 == 0) { printf("window %ld  p=%.3f  cue=%s\n", k, p, c->on ? "ON" : "off"); fflush(stdout); }
                    k++; nin = 0;
                }
            }
            off += PKT_SZ;
        }
        memmove(buf, buf + off, have - off); have -= off;
    }
    double wall = now_s(CLOCK_MONOTONIC) - w0, cpu = (double)clock() / CLOCKS_PER_SEC - cpu0, boot = now_s(CLOCK_BOOTTIME) - b0;
    close(s); fclose(o); fclose(sm);
    snprintf(fn, sizeof fn, "%s_stream_run.csv", pre); o = fopen(fn, "w");
    fprintf(o, "windows,sequence_errors,wall_s,cpu_s,boottime_s,peak_memory_mb\n%ld,%ld,%.6f,%.6f,%.6f,%.3f\n", k, seqerr, wall, cpu, boot, peak_rss_mb());
    fclose(o); fd_ctrl_free(c); free(win);
    printf("stream ended: %ld decisions, %ld sequence errors; cpu duty cycle %.2f %%\n", k, seqerr, 100 * cpu / wall);
    return 0;
}

static void usage(void) {
    fprintf(stderr, "usage: fd_edge bench MODEL REPLAY PREFIX [repeats] | loop MODEL REPLAY SECONDS PREFIX | realtime MODEL REPLAY MINUTES PREFIX"
                    " | dataset MODEL STREAMS OUT.csv | stream MODEL (listen|LAPTOP_IP) PREFIX\n");
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    if (argc >= 5 && !strcmp(argv[1], "bench")) return mode_bench(load(argv[2]), argv[3], argv[4], argc > 5 ? atoi(argv[5]) : 1);
    if (argc >= 6 && !strcmp(argv[1], "loop")) return mode_loop(load(argv[2]), argv[3], atof(argv[4]), argv[5]);
    if (argc >= 6 && !strcmp(argv[1], "realtime")) return mode_realtime(load(argv[2]), argv[3], atof(argv[4]), argv[5]);
    if (argc >= 5 && !strcmp(argv[1], "dataset")) return mode_dataset(load(argv[2]), argv[3], argv[4]);
    if (argc >= 5 && !strcmp(argv[1], "stream")) return mode_stream(load(argv[2]), argv[3], argv[4]);
    usage(); return 2;
}
