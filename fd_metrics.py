import numpy as np

import fd_config as C
from fd_data import runs, sequences, PRE, FOG, POST

CUE = (PRE, FOG, POST)


def hysteresis(p, on_th, off_th):
    out = np.zeros(len(p), bool); st = False
    for t, v in enumerate(p):
        st = (v >= off_th) if st else (v >= on_th)
        out[t] = st
    return out


def apply_hysteresis(p, meta, on_th, off_th, seqs=None):
    on = np.zeros(len(p), bool)
    for idx in (seqs if seqs is not None else sequences(meta)):
        on[idx] = hysteresis(p[idx], on_th, off_th)
    return on


class Evaluator:
    def __init__(self, meta, y4):
        self.meta = meta
        self.y2 = meta["y2"].to_numpy()
        self.target = np.isin(y4, CUE)
        self.seqs = list(sequences(meta))
        self.hours = len(meta) * C.WIN / C.FS / 3600
        self.episodes, self.eligible = [], []
        for q, idx in enumerate(self.seqs):
            prev_end = 0
            for s, e in runs(self.y2[idx] == 2):
                self.episodes.append((q, s, e))
                if s - prev_end >= C.PRE_W:
                    self.eligible.append((q, s))
                prev_end = e

    def on(self, p, th):
        return apply_hysteresis(p, self.meta, th[0], th[1], self.seqs)

    def evaluate(self, on):
        n_fa = 0
        for idx in self.seqs:
            o = on[idx]; n = len(idx); zone = np.zeros(n, bool)
            for s, e in runs(self.y2[idx] == 2):
                zone[max(0, s - C.PRE_W):min(n, e + C.POST_W)] = True
            for a, b in runs(o):
                if not zone[a:b].any():
                    n_fa += 1
        det = sum(bool(on[self.seqs[q]][s:e].any()) for q, s, e in self.episodes)
        tim = 0
        for q, s in self.eligible:
            o = on[self.seqs[q]]
            if o[s - 1]:
                st = s - 1
                while st > 0 and o[st - 1]:
                    st -= 1
                tim += (s - st) <= C.PRE_W
        t = self.target
        tn, fp = np.sum(~t & ~on), np.sum(~t & on)
        tp, fn = np.sum(t & on), np.sum(t & ~on)
        return {"false_alarms_per_hour": n_fa / self.hours, "coverage_%": 100 * det / max(len(self.episodes), 1),
                "timely_%": 100 * tim / max(len(self.eligible), 1), "cue_specificity": float(tn / max(tn + fp, 1)),
                "cue_sensitivity": float(tp / max(tp + fn, 1)), "cue_on_%": 100 * float(on.mean()),
                "episodes": len(self.episodes), "eligible_episodes": len(self.eligible), "hours": self.hours}

    def surrogate(self, on, n_shifts=None, seed=None, samples=False):
        n_shifts = C.N_SHIFTS if n_shifts is None else n_shifts
        seed = C.SEED if seed is None else seed
        rng = np.random.default_rng(seed)
        det = np.zeros(n_shifts); tim = np.zeros(n_shifts)
        by_q_ep, by_q_el = {}, {}
        for q, s, e in self.episodes:
            by_q_ep.setdefault(q, []).append((s, e))
        for q, s in self.eligible:
            by_q_el.setdefault(q, []).append(s)
        for q, idx in enumerate(self.seqs):
            n = len(idx)
            if n < 2 or (q not in by_q_ep and q not in by_q_el):
                if n >= 2:
                    rng.integers(1, n, size=n_shifts)
                continue
            k = rng.integers(1, n, size=n_shifts)
            o = on[idx].astype(np.int64)
            c = np.concatenate([[0], np.cumsum(np.concatenate([o, o]))])

            def window_sum(lo, hi):
                a = (lo[None, :] - k[:, None]) % n
                return c[a + (hi - lo)[None, :]] - c[a]

            if q in by_q_ep:
                se = np.array(by_q_ep[q]); det += (window_sum(se[:, 0], se[:, 1]) > 0).sum(1)
            if q in by_q_el:
                s = np.array(by_q_el[q])
                last = window_sum(s - 1, s) > 0
                lo = s - 1 - C.PRE_W
                full = np.zeros_like(last)
                ok = lo >= 0
                if ok.any():
                    full[:, ok] = window_sum(lo[ok], s[ok] - 1) == C.PRE_W
                tim += (last & ~full).sum(1)
        det = 100 * det / max(len(self.episodes), 1); tim = 100 * tim / max(len(self.eligible), 1)
        obs = self.evaluate(on)
        if samples:
            return det, tim
        return {"coverage_surrogate": float(det.mean()), "p_coverage": float((1 + np.sum(det >= obs["coverage_%"])) / (1 + n_shifts)),
                "timely_surrogate": float(tim.mean()), "p_timely": float((1 + np.sum(tim >= obs["timely_%"])) / (1 + n_shifts))}

    def shifted(self, on, k_per_seq):
        out = np.zeros_like(on)
        for q, idx in enumerate(self.seqs):
            out[idx] = np.roll(on[idx], k_per_seq[q]) if len(idx) > 1 else on[idx]
        return out
