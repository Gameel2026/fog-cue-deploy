"""Exploratory sensor-tilt check (added after the planned analyses; not pre-registered).

The lower-back accelerometer signal of each external dataset is rotated by 5, 10, 15 and 20
degrees about each body axis (vertical, mediolateral, anteroposterior) before the reference
chain. Cue decisions are compared window by window with the unrotated reference, and the
change in coverage and false alarms is reported. The frozen model and thresholds are unchanged.

Run from the repository root:  python fd_tilt.py
Output: results/tilt_sensitivity.csv
"""
import os
import numpy as np
import pandas as pd

import fd_config as C
import fd_common as U
import fd_model as M
from fd_data import stretches, filter_design
from fd_faults import windows, causal, past_rows, hysteresis
from fd_core import features_batch

ANGLES = [5, 10, 15, 20]
AXES = {"V": 0, "ML": 1, "AP": 2}


def rotate(acc, axis, deg):
    """rotate the [V, ML, AP] signal about one axis by deg degrees"""
    i, j = [k for k in range(3) if k != axis]
    t = np.deg2rad(deg)
    out = acc.copy()
    out[:, i] = np.cos(t) * acc[:, i] - np.sin(t) * acc[:, j]
    out[:, j] = np.sin(t) * acc[:, i] + np.cos(t) * acc[:, j]
    return out


def run(model, mu, sd, theta, n_past, acc, sos, zi):
    W = windows(causal(acc, sos, zi), C.WIN)
    Z = M.standardised(features_batch(W, C.FS, C.NFFT, C.MODE_RES, np.float64), mu, sd)
    p = M.prob(model, Z, past_rows(len(Z), n_past), np.arange(len(Z)))
    return p, hysteresis(p, theta)


def main():
    sos, zi = filter_design()
    rows = []
    for name in C.EXTERNAL:
        model, mu, sd, theta, n_past = M.load(U.prefix_for(name))
        pool = U.Pool([name]); ev = pool.evaluator(name); meta = pool.metas[name]
        conds = [("none", 0)] + [(a, d) for a in AXES for d in ANGLES]
        res = {}
        for ax, deg in conds:
            ps, ons = [], []
            for _, _, _, acc, _ in stretches(name):
                x = acc if ax == "none" else rotate(acc, AXES[ax], deg)
                p, on = run(model, mu, sd, theta, n_past, x, sos, zi)
                ps.append(p); ons.append(on)
            res[(ax, deg)] = (np.concatenate(ps), np.concatenate(ons))
            if len(res[(ax, deg)][1]) != len(meta):
                raise RuntimeError(f"{name}: window count mismatch")
        p0, on0 = res[("none", 0)]; m0 = ev.evaluate(on0)
        for (ax, deg), (p, on) in res.items():
            if ax == "none":
                continue
            m = ev.evaluate(on)
            pa = [100 * float(np.mean(on[g.index] == on0[g.index])) for _, g in meta.groupby("subject")]
            rows.append({"dataset": name, "axis": ax, "degrees": deg,
                         "decision_agreement_%": 100 * float(np.mean(on == on0)), "min_patient_agreement_%": float(np.min(pa)),
                         "max_abs_dp": float(np.max(np.abs(p - p0))),
                         "delta_coverage_%": m["coverage_%"] - m0["coverage_%"],
                         "delta_false_alarms_per_hour": m["false_alarms_per_hour"] - m0["false_alarms_per_hour"]})
            print(f"{name:10s} {ax:2s} {deg:2d} deg  agreement {rows[-1]['decision_agreement_%']:7.2f} %  "
                  f"dCov {rows[-1]['delta_coverage_%']:+6.1f}  dFA/h {rows[-1]['delta_false_alarms_per_hour']:+6.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT, "tilt_sensitivity.csv"), index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
