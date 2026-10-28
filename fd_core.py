import numpy as np

EPS = 1e-12
LOCO, FREEZE = (0.5, 3.0), (3.0, 8.0)


class Features:
    def __init__(self, fs, win, nfft, mode_res):
        self.fs, self.win, self.nfft, self.mode_res = float(fs), int(win), int(nfft), float(mode_res)
        self.f = np.arange(self.nfft // 2 + 1) * self.fs / self.nfft
        self.hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.win) / self.win)

    def stat(self, x):
        vals, cnt = np.unique(np.round(x / self.mode_res), return_counts=True)
        m = x.mean(); d = x - m; m2 = np.mean(d ** 2)
        a = np.abs(x) + EPS
        return [m, np.median(x), vals[cnt.argmax()] * self.mode_res, x.min(), x.max(), x.max() - x.min(),
                len(a) / np.sum(1.0 / a), x.std(ddof=1), x.var(ddof=1), np.mean(np.abs(d)),
                np.mean(np.abs(x - np.median(x))),
                np.mean(d ** 4) / m2 ** 2 if m2 > 0 else np.nan,
                np.mean(d ** 3) / m2 ** 1.5 if m2 > 0 else np.nan,
                np.sqrt(np.mean(x ** 2))]

    def spec(self, x):
        xw = (x - x.mean()) * self.hann
        P = np.abs(np.fft.rfft(xw, self.nfft)) ** 2 / (self.fs * np.sum(self.hann ** 2))
        P[1:-1] *= 2
        f = self.f; df_ = f[1] - f[0]
        bp = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() * df_
        loco, frz = bp(*LOCO), bp(*FREEZE)
        p = P / (P.sum() + EPS); mean_f = np.sum(f * p)
        mag = np.sqrt(P); mag = mag / (mag.sum() + EPS); centroid = np.sum(f * mag)
        sd = np.sqrt(np.sum((f - mean_f) ** 2 * p)) + EPS
        return [loco, frz, frz / (loco + EPS), frz + loco, mean_f,
                f[min(np.searchsorted(np.cumsum(P), 0.5 * P.sum()), len(f) - 1)], centroid,
                np.sum((f - mean_f) ** 4 * p) / sd ** 4, -np.sum(p * np.log2(p + EPS)) / np.log2(len(p)), f[np.argmax(P)]]

    def __call__(self, w):
        out = []
        for ax in range(w.shape[1]):
            out += self.stat(w[:, ax]) + self.spec(w[:, ax])
        v = np.asarray(out, float)
        v[~np.isfinite(v)] = 0.0
        return v


class SOSFilter:
    def __init__(self, sos, zi_unit):
        self.sos = np.asarray(sos, float); self.zi_unit = np.asarray(zi_unit, float); self.z = None

    def __call__(self, x):
        x = np.asarray(x, float)
        if self.z is None:
            self.z = self.zi_unit[:, :, None] * x[0][None, None, :]
        y = np.empty_like(x)
        for n in range(len(x)):
            v = x[n]
            for s, (b0, b1, b2, a0, a1, a2) in enumerate(self.sos):
                z = self.z[s]; out = b0 * v + z[0]
                z[0] = b1 * v - a1 * out + z[1]; z[1] = b2 * v - a2 * out
                v = out
            y[n] = v
        return y


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class EdgeLSTM:
    def __init__(self, path):
        d = np.load(path)
        get = lambda k: (d[k].astype(np.float32) * d[k + "__scale"][:, None]) if (k + "__scale") in d.files else d[k].astype(np.float32)
        self.layers = [(get(f"weight_ih_l{l}"), get(f"weight_hh_l{l}"), (d[f"bias_ih_l{l}"] + d[f"bias_hh_l{l}"]).astype(np.float32))
                       for l in range(2)]
        self.Wh, self.bh = get("head_weight"), d["head_bias"].astype(np.float32)
        self.mu, self.sd = d["mu"].astype(np.float64), d["sd"].astype(np.float64)
        self.sos, self.zi = d["sos"], d["zi_unit"]
        self.theta = (float(d["theta_on"]), float(d["theta_off"]))
        self.n_past = int(d["n_past"])
        self.fs, self.win, self.nfft, self.mode_res = float(d["fs"]), int(d["win"]), int(d["nfft"]), float(d["mode_res"])
        self.H = self.layers[0][1].shape[1]

    def prob(self, Z):
        x = Z
        for Wi, Wh, b in self.layers:
            h = np.zeros(self.H, np.float32); c = np.zeros(self.H, np.float32); hs = []
            xi = x @ Wi.T + b
            for t in range(len(x)):
                g = xi[t] + Wh @ h
                i, f, gg, o = np.split(g, 4)
                c = _sig(f) * c + _sig(i) * np.tanh(gg); h = _sig(o) * np.tanh(c); hs.append(h)
            x = np.stack(hs)
        z = self.Wh @ x[-1] + self.bh; z = z - z.max(); e = np.exp(z)
        return float(e[1] / e.sum())


class CueController:
    def __init__(self, model):
        self.m = model; self.filt = SOSFilter(model.sos, model.zi)
        self.feat = Features(model.fs, model.win, model.nfft, model.mode_res)
        self.buf = []; self.n_past = model.n_past; self.on = False

    def step(self, raw_window):
        w = self.filt(raw_window)
        f = self.feat(w)
        z = np.clip((f - self.m.mu) / self.m.sd, -10, 10).astype(np.float32)
        self.buf.append(z); self.buf = self.buf[-self.n_past:]
        Z = np.zeros((self.n_past, len(z)), np.float32); Z[self.n_past - len(self.buf):] = np.stack(self.buf)
        p = self.m.prob(Z)
        a, b = self.m.theta
        self.on = (p >= b) if self.on else (p >= a)
        return p, self.on, f


def lstm_batch(layers, Wh, bh, Z, act_bits=None):
    x = np.asarray(Z, np.float32)
    B, T, _ = x.shape
    for Wi, Whh, b in layers:
        H = Whh.shape[1]
        h = np.zeros((B, H), np.float32); c = np.zeros((B, H), np.float32); hs = np.empty((B, T, H), np.float32)
        for t in range(T):
            xt = x[:, t]
            if act_bits:
                xt = quantise_rows(xt, act_bits); hq = quantise_rows(h, act_bits)
            else:
                hq = h
            g = xt @ Wi.T + hq @ Whh.T + b
            i, f, gg, o = np.split(g, 4, axis=1)
            c = _sig(f) * c + _sig(i) * np.tanh(gg); h = (_sig(o) * np.tanh(c)).astype(np.float32); hs[:, t] = h
        x = hs
    z = x[:, -1] @ Wh.T + bh
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return (e[:, 1] / e.sum(1)).astype(np.float64)


def quantise_rows(a, bits):
    q = 2 ** (bits - 1) - 1
    s = np.maximum(np.abs(a).max(axis=-1, keepdims=True), 1e-12) / q
    return (np.clip(np.round(a / s), -q, q) * s).astype(np.float32)


def features_batch(W, fs, nfft, mode_res, dtype=np.float64):
    W = np.asarray(W, dtype)
    N, L, A = W.shape
    f = (np.arange(nfft // 2 + 1) * fs / nfft).astype(dtype)
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)).astype(dtype)
    out = np.empty((N, 24 * A), dtype)
    eps = dtype(EPS) if dtype is not np.float64 else EPS
    for ax in range(A):
        x = W[:, :, ax]
        m = x.mean(1, keepdims=True); d = x - m; m2 = np.mean(d ** 2, 1)
        med = np.median(x, 1)
        r = np.sort(np.round(x / dtype(mode_res)), 1)
        counts = (r[:, :, None] == r[:, None, :]).sum(2)
        mode = r[np.arange(N), counts.argmax(1)] * dtype(mode_res)
        a = np.abs(x) + eps
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = np.where(m2 > 0, np.mean(d ** 4, 1) / m2 ** 2, np.nan)
            skew = np.where(m2 > 0, np.mean(d ** 3, 1) / m2 ** 1.5, np.nan)
        stat = [m[:, 0], med, mode, x.min(1), x.max(1), x.max(1) - x.min(1), L / np.sum(1.0 / a, 1),
                x.std(1, ddof=1), x.var(1, ddof=1), np.mean(np.abs(d), 1), np.mean(np.abs(x - med[:, None]), 1),
                kurt, skew, np.sqrt(np.mean(x ** 2, 1))]
        xw = d * hann
        P = (np.abs(np.fft.rfft(xw, nfft, axis=1)) ** 2 / (fs * np.sum(hann ** 2))).astype(dtype)
        P[:, 1:-1] *= 2
        df_ = f[1] - f[0]
        band = lambda lo, hi: P[:, (f >= lo) & (f < hi)].sum(1) * df_
        loco, frz = band(*LOCO), band(*FREEZE)
        tot = P.sum(1)
        p = P / (tot[:, None] + eps); mean_f = (f * p).sum(1)
        mag = np.sqrt(P); mag = mag / (mag.sum(1, keepdims=True) + eps)
        sd = np.sqrt(((f[None] - mean_f[:, None]) ** 2 * p).sum(1)) + eps
        cs = np.cumsum(P, 1)
        medf = f[np.minimum((cs < 0.5 * tot[:, None]).sum(1), len(f) - 1)]
        spec = [loco, frz, frz / (loco + eps), frz + loco, mean_f, medf, (f * mag).sum(1),
                ((f[None] - mean_f[:, None]) ** 4 * p).sum(1) / sd ** 4, -(p * np.log2(p + eps)).sum(1) / np.log2(P.shape[1]), f[P.argmax(1)]]
        out[:, 24 * ax:24 * ax + 24] = np.column_stack(stat + spec)
    out = out.astype(np.float64)
    out[~np.isfinite(out)] = 0.0
    return out
