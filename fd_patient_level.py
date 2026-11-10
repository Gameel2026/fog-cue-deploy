import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import fd_config as C
import fd_common as U
from fd_metrics import Evaluator


def per_patient(name, meta, y4, on):
    rows = []
    for s, g in meta.groupby("subject", sort=True):
        idx = g.index.to_numpy()
        ev = Evaluator(meta.loc[idx].reset_index(drop=True), y4[idx])
        r = ev.evaluate(on[idx])
        if r["episodes"] > 0:
            r.update(ev.surrogate(on[idx], n_shifts=C.N_SHIFTS))
        r["subject"] = str(s); r["dataset"] = name
        rows.append(r)
    return rows


def main():
    frozen = json.load(open(os.path.join(C.OUT, "frozen.json")))
    cand = frozen["candidate"]; th = {k: tuple(v) for k, v in frozen["thetas"].items()}
    rows = []
    dev = U.Pool(C.DEVELOPMENT)
    oof = np.load(os.path.join(C.OUT, "cv", f"{cand}_oof.npy"))
    for n in C.DEVELOPMENT:
        s = dev.slices[n]; ev = dev.evaluator(n)
        on = ev.on(oof[s], th[C.CONTEXT_OF[n]])
        rows += per_patient(n, dev.metas[n], dev.y4[s], on)
        print(n, "done", flush=True)
    for n in C.EXTERNAL:
        pool = U.Pool([n]); meta = pool.metas[n]
        w = pd.read_csv(os.path.join(C.OUT, f"external_{n}_windows.csv"))
        if len(w) != len(meta):
            raise RuntimeError(f"{n}: window count mismatch")
        rows += per_patient(n, meta, pool.y4, w["cue_on"].to_numpy().astype(bool))
        print(n, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT, "patient_level.csv"), index=False)
    summ = []
    for n, g in df.groupby("dataset", sort=False):
        e = g[g["episodes"] > 0]
        cov, ch = e["coverage_%"].to_numpy(), e["coverage_surrogate"].to_numpy()
        d = cov - ch
        try:
            p_w = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else np.nan
        except ValueError:
            p_w = np.nan
        q = lambda x, k: float(np.percentile(x, k))
        summ.append({"dataset": n, "patients": len(g), "patients_with_episodes": len(e),
                     "patients_with_one_covered_episode": int((e["coverage_%"] > 0).sum()),
                     "coverage_median": q(cov, 50), "coverage_q1": q(cov, 25), "coverage_q3": q(cov, 75),
                     "gain_over_chance_median": q(d, 50), "gain_q1": q(d, 25), "gain_q3": q(d, 75),
                     "patients_above_own_chance": int((d > 0).sum()),
                     "patients_p_below_0.05": int((e["p_coverage"] < 0.05).sum()), "wilcoxon_p_gain": p_w,
                     "fa_per_h_median": q(g["false_alarms_per_hour"], 50), "fa_q1": q(g["false_alarms_per_hour"], 25),
                     "fa_q3": q(g["false_alarms_per_hour"], 75), "fa_max": float(g["false_alarms_per_hour"].max())})
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(C.OUT, "patient_level_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
