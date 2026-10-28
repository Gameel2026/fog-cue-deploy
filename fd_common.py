import os
import json
import hashlib
import platform
import numpy as np
import pandas as pd

import fd_config as C
from fd_data import build, past_index
from fd_metrics import Evaluator, CUE


def quick_mode():
    C.OUT = "results_quick"
    C.N_FOLDS = 2
    C.MAX_EPOCHS = 1
    C.N_SHIFTS = 50


class Pool:
    def __init__(self, names):
        self.names = list(names)
        parts = [build(n) for n in self.names]
        self.F = np.vstack([p[0] for p in parts])
        self.y4 = np.concatenate([p[2] for p in parts])
        self.y = np.isin(self.y4, CUE).astype(np.int64)
        self.metas, self.slices, start = {}, {}, 0
        ds, subj = [], []
        for n, (F, meta, _) in zip(self.names, parts):
            self.metas[n] = meta.reset_index(drop=True)
            self.slices[n] = slice(start, start + len(F))
            ds += [n] * len(F); subj += list(meta["subject"].astype(str))
            start += len(F)
        self.dataset = np.array(ds)
        self.groups = np.array([f"{a}:{b}" for a, b in zip(ds, subj)])
        self._P = {}
        self._ev = {}

    def P(self, n_past):
        if n_past not in self._P:
            blocks = []
            for n in self.names:
                p = past_index(self.metas[n], n_past)
                blocks.append(np.where(p >= 0, p + self.slices[n].start, -1))
            self._P[n_past] = np.vstack(blocks)
        return self._P[n_past]

    def evaluator(self, name):
        if name not in self._ev:
            s = self.slices[name]
            self._ev[name] = Evaluator(self.metas[name], self.y4[s])
        return self._ev[name]


def model_prefix(context):
    return os.path.join(C.OUT, "model" if context == "home" else f"model_{context}")


def prefix_for(name):
    return model_prefix(C.CONTEXT_OF[name])


def grid(ev, p):
    rows = []
    for th in C.GRID:
        r = ev.evaluate(ev.on(p, th)); r["theta_on"], r["theta_off"] = th; rows.append(r)
    return pd.DataFrame(rows)


def operating_point(table):
    t = table.reset_index(drop=True)
    ok = t[(t["false_alarms_per_hour"] <= C.FA_TARGET) & (t["cue_specificity"] >= C.SPEC_FLOOR)]
    if len(ok):
        best = ok.sort_values(["coverage_%", "timely_%", "false_alarms_per_hour"], ascending=[False, False, True], kind="stable").iloc[0]
        return best, "primary"
    ok = t[t["cue_specificity"] >= C.SPEC_FLOOR]
    if len(ok):
        best = ok.sort_values(["false_alarms_per_hour", "coverage_%"], ascending=[True, False], kind="stable").iloc[0]
        return best, "fallback_min_fa"
    best = t.sort_values(["cue_specificity", "coverage_%"], ascending=[False, False], kind="stable").iloc[0]
    return best, "fallback_max_specificity"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["numpy", "scipy", "pandas", "sklearn", "torch"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def config_snapshot():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(C).items()
            if k.isupper() and k != "GRID"}


def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
