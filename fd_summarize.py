import os
import sys
import glob
import json
import struct
import numpy as np
import pandas as pd

import fd_config as C
import fd_common as U

if "--quick" in sys.argv:
    U.quick_mode()

from fd_core import EdgeLSTM, CueController


def q(s):
    s = pd.Series(s).astype(float)
    return {"median": float(s.median()), "p95": float(s.quantile(0.95)), "p99": float(s.quantile(0.99)), "max": float(s.max())}


def hysteresis_groups(p, groups, theta):
    on = np.zeros(len(p), bool); st = False; last = None
    for i, (v, g) in enumerate(zip(p, groups)):
        if g != last:
            st = False; last = g
        st = (v >= theta[1]) if st else (v >= theta[0]); on[i] = st
    return on


def bench(path):
    b = pd.read_csv(path)
    return {"windows_per_pass": int((b["pass"] == 0).sum()), "passes": int(b["pass"].nunique()), "total_ms": q(b["total_ms"]),
            "filter_ms": q(b["filter_ms"]), "features_ms": q(b["features_ms"]), "lstm_ms": q(b["lstm_ms"]),
            "max_abs_dp_vs_python": float(np.nanmax(np.abs(b["p"] - b["p_ref"])))}


def dataset(path, name, theta):
    theta = tuple(theta[C.CONTEXT_OF[name]])
    d = pd.read_csv(path)
    ref = np.load(os.path.join(C.OUT, f"stream_p_{name}.npy"))
    if len(ref) != len(d):
        raise RuntimeError(f"{path}: {len(d)} windows, python reference has {len(ref)}")
    pool = U.Pool([name]); meta = pool.metas[name]
    on_ref = pool.evaluator(name).on(ref, theta)
    on_c = d["cue_on"].to_numpy().astype(bool)
    on_c_check = hysteresis_groups(d["p"].to_numpy(), d["stretch"].to_numpy(), theta)
    pa = [100 * float(np.mean(on_c[g.index] == on_ref[g.index])) for _, g in meta.groupby("subject")]
    out = {"windows": int(len(d)), "patients": len(pa), "max_abs_dp_vs_python": float(np.max(np.abs(d["p"].to_numpy() - ref))),
           "decision_agreement_%": 100 * float(np.mean(on_c == on_ref)), "disagreements": int(np.sum(on_c != on_ref)),
           "min_patient_agreement_%": float(np.min(pa)), "internal_hysteresis_consistent": bool(np.all(on_c == on_c_check))}
    try:
        import fd_model as M
        model, mu, sd, th, n_past = M.load(U.prefix_for(name))
        pt = M.prob(model, M.standardised(pool.F, mu, sd), pool.P(n_past), np.arange(len(pool.F)))
        on_t = pool.evaluator(name).on(pt, theta)
        out["max_abs_dp_vs_torch"] = float(np.max(np.abs(d["p"].to_numpy() - pt)))
        out["decision_agreement_vs_torch_%"] = 100 * float(np.mean(on_c == on_t))
    except ImportError:
        pass
    return out


def realtime(path):
    r = pd.read_csv(path); run = pd.read_csv(path.replace("_realtime.csv", "_realtime_run.csv")).iloc[0]
    d = r["delay_monotonic_ms"]
    late = r[d > 500]
    return {"decisions": int(len(r)), "delay_ms": q(d), "processing_ms": q(r["processing_ms"]),
            "within_100ms_%": 100 * float(np.mean(d <= 100)), "over_500ms": int(len(late)),
            "first_late_window": int(late["window"].iloc[0]) if len(late) else None,
            "max_boottime_minus_monotonic_ms": float((r["delay_boottime_ms"] - r["delay_monotonic_ms"]).max()),
            "wall_s": float(run["wall_s"]), "cpu_s": float(run["cpu_s"]), "duty_cycle_%": 100 * float(run["cpu_s"] / run["wall_s"]),
            "suspended_s": float(run["boottime_s"] - run["wall_s"]), "peak_memory_mb": float(run["peak_memory_mb"])}


def stream(path, npz, sender=None):
    s = pd.read_csv(path); run = pd.read_csv(path.replace("_stream.csv", "_stream_run.csv")).iloc[0]
    model = EdgeLSTM(npz)
    raw = np.fromfile(path.replace("_stream.csv", "_stream_samples.bin"), np.float64).reshape(-1, model.win, 3)
    c = CueController(model); p = np.array([c.step(w)[0] for w in raw])
    n = min(len(p), len(s))
    out = {"decisions": int(len(s)), "sequence_errors": int(run["sequence_errors"]), "processing_ms": q(s["processing_ms"]),
           "on_phone_delay_ms": q(s["on_phone_delay_ms"]), "duty_cycle_%": 100 * float(run["cpu_s"] / run["wall_s"]),
           "max_abs_dp_vs_python": float(np.max(np.abs(s["p"].to_numpy()[:n] - p[:n])))}
    if sender and os.path.exists(sender):
        sd = pd.read_csv(sender)
        out["round_trip_ms"] = q(sd["round_trip_ms"].dropna())
        out["decisions_received_by_laptop"] = int(len(sd))
    return out


def main():
    d = sys.argv[1]
    frozen = json.load(open(os.path.join(C.OUT, "frozen.json")))
    theta = frozen["thetas"]
    npz = os.path.join(C.OUT, "model_fp32.npz")
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "*_bench.csv"))):
        res[os.path.basename(f)] = bench(f)
    for f in sorted(glob.glob(os.path.join(d, "*_dataset_*.csv"))):
        name = os.path.splitext(os.path.basename(f))[0].split("_dataset_")[1]
        res[os.path.basename(f)] = dataset(f, name, theta)
    for f in sorted(glob.glob(os.path.join(d, "*_realtime.csv"))):
        res[os.path.basename(f)] = realtime(f)
    for f in sorted(glob.glob(os.path.join(d, "*_stream.csv"))):
        sender = f.replace("_stream.csv", "_sender.csv")
        res[os.path.basename(f)] = stream(f, npz, sender)
    U.write_json(res, os.path.join(d, "summary.json"))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
