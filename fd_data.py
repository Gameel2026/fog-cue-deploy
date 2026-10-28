import os
import pickle
import numpy as np
import pandas as pd
from scipy import signal, stats

import fd_config as C

NOFOG, PRE, FOG, POST = 0, 1, 2, 3
EPS = 1e-12
FEATURE_NAMES = [f"{k}_{ax}" for ax in (1, 2, 3) for k in
                 ["mean", "median", "mode", "min", "max", "range", "harmean", "StDev", "Variance", "Mean_Ab", "Med_Ab",
                  "kurtosis", "skewness", "RMS", "Locomotorypower", "Freezepower", "Freezeindex", "Sumpower", "Meanf",
                  "medf", "spectralcentroid", "spectralKurtosis", "spectralEntropy", "Spectralpeak"]]


def runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def filter_design():
    sos = signal.butter(C.FILT_ORDER, C.BAND, btype="bandpass", fs=C.FS, output="sos")
    return sos, signal.sosfilt_zi(sos)


def merge_gaps(lab):
    lab = lab.copy()
    for a, b in runs(lab != 0):
        seg = lab[a:b]
        eps = runs(seg == 2)
        for (s1, e1), (s2, e2) in zip(eps[:-1], eps[1:]):
            if (s2 - e1) / C.FS < C.MERGE_GAP_S:
                seg[e1:s2] = 2
        lab[a:b] = seg
    return lab


def recordings(name):
    path = os.path.join(C.PREP, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run fd_prepare.py first")
    with open(path, "rb") as f:
        recs = pickle.load(f)
    for r in recs:
        lab = r["label"].astype(int)
        if C.DATASETS[name]:
            lab = merge_gaps(lab)
        yield r["subject"], r["recording"], np.asarray(r["acc"], float), lab, r.get("severity")


def stretches(name, with_severity=False):
    for subj, rec, acc, lab, sev in recordings(name):
        seg = 0
        for a, b in runs(lab != 0):
            if b - a < 3 * C.WIN:
                continue
            n = (b - a) // C.WIN
            item = (subj, rec, seg, acc[a:a + n * C.WIN], lab[a:a + n * C.WIN])
            if with_severity:
                item = item + (None if sev is None else sev[a:a + n * C.WIN],)
            yield item
            seg += 1


def stat_features(x):
    vals, cnt = np.unique(np.round(x / C.MODE_RES), return_counts=True)
    return [x.mean(), np.median(x), vals[cnt.argmax()] * C.MODE_RES, x.min(), x.max(), np.ptp(x), stats.hmean(np.abs(x) + EPS),
            x.std(ddof=1), x.var(ddof=1), np.mean(np.abs(x - x.mean())), np.mean(np.abs(x - np.median(x))),
            stats.kurtosis(x, fisher=False), stats.skew(x), np.sqrt(np.mean(x ** 2))]


def spectral_features(x):
    f, P = signal.periodogram(x, fs=C.FS, nfft=C.NFFT, window="hann", detrend="constant")
    df_ = f[1] - f[0]
    bp = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() * df_
    loco, frz = bp(0.5, 3.0), bp(3.0, 8.0)
    p = P / (P.sum() + EPS)
    mean_f = np.sum(f * p)
    mag = np.sqrt(P); mag = mag / (mag.sum() + EPS)
    sd = np.sqrt(np.sum((f - mean_f) ** 2 * p)) + EPS
    return [loco, frz, frz / (loco + EPS), frz + loco, mean_f, f[min(np.searchsorted(np.cumsum(P), 0.5 * P.sum()), len(f) - 1)],
            np.sum(f * mag), np.sum((f - mean_f) ** 4 * p) / sd ** 4, -np.sum(p * np.log2(p + EPS)) / np.log2(len(p)), f[np.argmax(P)]]


def relabel(y2, n):
    y4 = np.where(y2 == 2, FOG, NOFOG)
    eps = runs(y2 == 2); pre_start = []
    for m, (s, _) in enumerate(eps):
        prev_end = eps[m - 1][1] if m else 0
        pre = min(C.PRE_W, s - prev_end)
        y4[s - pre:s] = PRE; pre_start.append(s - pre)
    for m, (_, e) in enumerate(eps):
        limit = pre_start[m + 1] if m + 1 < len(eps) else n
        y4[e:e + min(C.POST_W, limit - e)] = POST
    return y4


def _params(name):
    p = {k: getattr(C, k) for k in ["FS", "WIN", "NFFT", "FOG_MIN_SAMPLES", "MERGE_GAP_S", "BAND", "FILT_ORDER", "PRE_W", "POST_W", "MODE_RES"]}
    src = os.path.join(C.PREP, f"{name}.pkl")
    p["source"] = (os.path.getsize(src), int(os.path.getmtime(src))) if os.path.exists(src) else None
    return p


def build(name):
    os.makedirs(C.PREP, exist_ok=True)
    path = os.path.join(C.PREP, f"features_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if cached["params"] == _params(name):
            return cached["F"], cached["meta"], cached["y4"]
    sos, zi = filter_design()
    F, meta, y4s = [], [], []
    for subj, rec, seg, acc, lab, sev in stretches(name, with_severity=True):
        xf, _ = signal.sosfilt(sos, acc, axis=0, zi=zi[:, :, None] * acc[0][None, None, :])
        n = len(acc) // C.WIN
        y2 = np.array([2 if np.sum(lab[k * C.WIN:(k + 1) * C.WIN] == 2) >= C.FOG_MIN_SAMPLES else 1 for k in range(n)])
        for k in range(n):
            w = xf[k * C.WIN:(k + 1) * C.WIN]
            F.append([v for ax in range(3) for v in stat_features(w[:, ax]) + spectral_features(w[:, ax])])
            sv = 0
            if sev is not None and y2[k] == 2:
                vals = sev[k * C.WIN:(k + 1) * C.WIN]; vals = vals[vals > 0]
                sv = int(np.bincount(vals).argmax()) if len(vals) else 0
            meta.append((subj, rec, seg, k, y2[k], sv))
        y4s.append(relabel(y2, n))
    F = np.asarray(F, float); F[~np.isfinite(F)] = 0.0
    meta = pd.DataFrame(meta, columns=["subject", "recording", "seg", "pos", "y2", "severity"])
    y4 = np.concatenate(y4s)
    with open(path, "wb") as f:
        pickle.dump({"params": _params(name), "F": F, "meta": meta, "y4": y4}, f, protocol=4)
    return F, meta, y4


def raw_stretches(name):
    return [s[3] for s in stretches(name)]


def sequences(meta):
    for _, g in meta.groupby(["subject", "recording", "seg"], sort=True):
        yield g.sort_values("pos").index.to_numpy()


def past_index(meta, n):
    P = np.full((len(meta), n), -1, dtype=np.int64)
    for idx in sequences(meta):
        for j, i in enumerate(idx):
            for k in range(n):
                src = j - (n - 1) + k
                if src >= 0:
                    P[i, k] = idx[src]
    return P


def gather(A, P):
    out = A[np.maximum(P, 0)]
    out[P < 0] = 0
    return out
