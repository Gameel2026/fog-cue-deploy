import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import fd_config as C
import fd_common as U

if "--quick" in sys.argv:
    U.quick_mode()

import fd_model as M
from fd_metrics import Evaluator


def reference_probabilities(pool, prefix):
    model, mu, sd, theta, n_past = M.load(prefix)
    Zs = M.standardised(pool.F, mu, sd)
    return M.prob(model, Zs, pool.P(n_past), np.arange(len(pool.F))), theta


def main():
    for name in C.EXTERNAL:
        external(name)


def external(name):
    pool = U.Pool([name])
    p, theta = reference_probabilities(pool, U.prefix_for(name))
    ev = pool.evaluator(name)
    on = ev.on(p, theta)
    other = {}
    for ctx in C.CONTEXTS:
        if ctx != C.CONTEXT_OF[name]:
            th = M.load(U.model_prefix(ctx))[3]
            r = ev.evaluate(ev.on(p, th))
            other[f"with_{ctx}_threshold"] = {k: r[k] for k in ["false_alarms_per_hour", "coverage_%", "timely_%", "cue_specificity"]}
    res = {**ev.evaluate(on), **ev.surrogate(on, n_shifts=C.N_SHIFTS), "auc": roc_auc_score(pool.y, p),
           "theta_on": theta[0], "theta_off": theta[1], "context": C.CONTEXT_OF[name],
           "patients": int(len(np.unique(pool.groups))), **other}
    U.write_json(res, os.path.join(C.OUT, f"external_{name}_summary.json"))
    meta = pool.metas[name]
    w = meta.copy(); w["y4"] = pool.y4; w["p"] = p; w["cue_on"] = on.astype(int)
    w.to_csv(os.path.join(C.OUT, f"external_{name}_windows.csv"), index=False)
    rows = []
    for s, g in meta.groupby("subject", sort=True):
        idx = g.index.to_numpy()
        sub = Evaluator(meta.loc[idx].reset_index(drop=True), pool.y4[idx])
        r = sub.evaluate(on[idx]); r["subject"] = s
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, f"external_{name}_per_patient.csv"), index=False)
    if "severity" in meta and (meta["severity"] > 0).any():
        sev = meta["severity"].to_numpy(); by = {}
        for q, s0, e0 in ev.episodes:
            idx = ev.seqs[q]
            v = sev[idx[s0:e0]]; v = v[v > 0]
            k = int(np.bincount(v).argmax()) if len(v) else 0
            d = by.setdefault(k, [0, 0]); d[0] += 1; d[1] += int(on[idx[s0:e0]].any())
        pd.DataFrame([{"severity": k, "episodes": v[0], "detected": v[1], "coverage_%": 100 * v[1] / v[0]} for k, v in sorted(by.items())]
                     ).to_csv(os.path.join(C.OUT, f"external_{name}_by_severity.csv"), index=False)
    print(name)
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
