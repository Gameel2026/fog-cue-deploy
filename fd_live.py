import os
import re
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import fd_config as C
from fd_core import EdgeLSTM, CueController


def segments(run_dir):
    ev = pd.read_csv(os.path.join(run_dir, "events.csv"))
    segs, cur = [], None
    for t, e in zip(ev["elapsed_ns"], ev["event"]):
        m = re.match(r"input rate=([\d.]+) path=(\w+) first_sample_after_ns=(\d+)", str(e))
        if m:
            cur = {"rate": float(m.group(1)), "path": m.group(2), "after_ns": int(m.group(3)), "start_ns": int(t), "end_ns": None}
            segs.append(cur)
        elif str(e) in ("sensor_off",) or str(e).startswith("finish") or str(e).startswith("stopped"):
            if cur is not None and cur["end_ns"] is None:
                cur["end_ns"] = int(t)
    for s in segs:
        if s["end_ns"] is None:
            s["end_ns"] = np.iinfo(np.int64).max
    return segs, ev


def to_model_frame(raw):
    x = raw["x"].astype(str).map(np.float32).to_numpy(np.float64)
    y = raw["y"].astype(str).map(np.float32).to_numpy(np.float64)
    z = raw["z"].astype(str).map(np.float32).to_numpy(np.float64)
    return np.column_stack([-y / C.G, x / C.G, -z / C.G])


def resample_like_device(t, s, fs):
    period = int(round(1e9 / fs))
    out_v, out_src = [s[0]], [int(t[0])]
    nxt = int(t[0]) + period
    pt, pv = int(t[0]), s[0]
    for i in range(1, len(t)):
        ti = int(t[i])
        if ti <= pt:
            continue
        while nxt <= ti:
            w = (nxt - pt) / (ti - pt)
            out_v.append(pv + w * (s[i] - pv)); out_src.append(ti); nxt += period
        pt, pv = ti, s[i]
    return np.array(out_v), np.array(out_src, dtype=np.int64)


def analyse(run_dir, npz):
    model = EdgeLSTM(npz)
    raw = pd.read_csv(os.path.join(run_dir, "raw.csv"), dtype={"x": str, "y": str, "z": str})
    dec = pd.read_csv(os.path.join(run_dir, "decisions.csv"))
    segs, _ = segments(run_dir)
    rows, all_dec = [], []
    for k, sg in enumerate(segs):
        r = raw[(raw["arrival_ns"] >= sg["start_ns"] - 5_000_000_000) & (raw["arrival_ns"] <= sg["end_ns"])]
        r = r[r["event_ns"] > sg["after_ns"]]
        t = r["event_ns"].to_numpy(np.int64)
        s = to_model_frame(r)
        if sg["path"] == "resampled":
            s, src = resample_like_device(t, s, model.fs)
        else:
            src = t
        d = dec[(dec["decision_ns"] >= sg["start_ns"]) & (dec["decision_ns"] <= sg["end_ns"])].reset_index(drop=True)
        c = CueController(model)
        n = min(len(s) // model.win, len(d))
        p_off, on_off = np.zeros(n), np.zeros(n, bool)
        for j in range(n):
            p_off[j], on_off[j], _ = c.step(s[j * model.win:(j + 1) * model.win])
        d = d.iloc[:n]
        dt = np.diff(t) / 1e6
        rows.append({"segment": k, "path": sg["path"], "input_rate_hz": sg["rate"], "windows": int(n),
                     "max_abs_dp": float(np.max(np.abs(d["p"].to_numpy() - p_off))) if n else np.nan,
                     "decision_agreement_%": 100 * float(np.mean(d["cue_on"].to_numpy().astype(bool) == on_off)) if n else np.nan,
                     "source_sample_match": bool(np.all(d["last_sample_ns"].to_numpy() == src[np.arange(1, n + 1) * model.win - 1])) if n else False,
                     "delay_median_ms": float(d["delay_ms"].median()), "delay_p95_ms": float(d["delay_ms"].quantile(0.95)),
                     "delay_p99_ms": float(d["delay_ms"].quantile(0.99)), "delay_max_ms": float(d["delay_ms"].max()),
                     "within_100ms_%": 100 * float(np.mean(d["delay_ms"] <= 100)), "over_500ms": int(np.sum(d["delay_ms"] > 500)),
                     "processing_median_ms": float(d["processing_ms"].median()), "processing_p99_ms": float(d["processing_ms"].quantile(0.99)),
                     "sample_interval_median_ms": float(np.median(dt)), "sample_interval_sd_ms": float(np.std(dt)),
                     "sample_gaps_over_1p5x": int(np.sum(dt > 1.5 * np.median(dt))), "cue_on_%": 100 * float(d["cue_on"].mean()) if n else np.nan,
                     "p_sd": float(d["p"].std()) if n else np.nan})
    return pd.DataFrame(rows)


def energy(run_dir, capacity_mah):
    b = pd.read_csv(os.path.join(run_dir, "battery.csv"))
    ev = pd.read_csv(os.path.join(run_dir, "events.csv"))
    loops = [int(e.split()[1]) for e in ev["event"].astype(str) if e.startswith("loop_decisions")]
    sign = -1.0 if b.loc[b["block"] == "idle", "current_ua"].median() < 0 else 1.0
    b["discharge_ma"] = sign * b["current_ua"] / 1000.0
    starts = ev[ev["event"].astype(str).str.startswith("block_start")]
    blocks = []
    edges = list(starts["elapsed_ns"]) + [int(b["elapsed_ns"].max()) + 1]
    for i, (t0, e) in enumerate(zip(starts["elapsed_ns"], starts["event"])):
        g = b[(b["elapsed_ns"] >= t0 + 60_000_000_000) & (b["elapsed_ns"] < edges[i + 1])]
        row = {"block": i, "kind": e.split()[1], "seconds_used": len(g), "mean_ma": g["discharge_ma"].mean(),
               "sd_ma": g["discharge_ma"].std(), "mean_v": g["voltage_mv"].mean() / 1000.0, "screen_on_fraction": g["screen_on"].mean()}
        if "charge_counter_uah" in g and len(g) > 60:
            t_h = (g["elapsed_ns"].to_numpy() - g["elapsed_ns"].iloc[0]) / 3.6e12
            row["counter_ma"] = float(-np.polyfit(t_h, g["charge_counter_uah"].to_numpy(float), 1)[0] / 1000.0)
        if "plugged" in g:
            row["plugged_fraction"] = float((g["plugged"] > 0).mean())
        blocks.append(row)
    bl = pd.DataFrame(blocks)
    out = {"blocks": bl.to_dict(orient="records")}
    pairs = {"live": [], "loop": []}
    for i in range(1, len(bl)):
        if bl.loc[i, "kind"] in pairs and bl.loc[i - 1, "kind"] == "idle":
            nxt = bl.loc[i + 1, "mean_ma"] if i + 1 < len(bl) and bl.loc[i + 1, "kind"] == "idle" else bl.loc[i - 1, "mean_ma"]
            pairs[bl.loc[i, "kind"]].append(bl.loc[i, "mean_ma"] - 0.5 * (bl.loc[i - 1, "mean_ma"] + nxt))
    if "counter_ma" in bl:
        cp = {"live": [], "loop": []}
        for i in range(1, len(bl)):
            if bl.loc[i, "kind"] in cp and bl.loc[i - 1, "kind"] == "idle":
                nxt = bl.loc[i + 1, "counter_ma"] if i + 1 < len(bl) and bl.loc[i + 1, "kind"] == "idle" else bl.loc[i - 1, "counter_ma"]
                cp[bl.loc[i, "kind"]].append(bl.loc[i, "counter_ma"] - 0.5 * (bl.loc[i - 1, "counter_ma"] + nxt))
        out["charge_counter_check"] = {f"{k}_minus_idle_ma": float(np.mean(v)) for k, v in cp.items() if len(v)}
    for k, v in pairs.items():
        v = np.array(v)
        if len(v):
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            q = student_t.ppf(0.975, len(v) - 1) if len(v) > 1 else np.nan
            out[f"{k}_minus_idle_ma"] = {"mean": float(v.mean()), "ci95": [float(v.mean() - q * se), float(v.mean() + q * se)], "n": int(len(v))}
    if loops and out.get("loop_minus_idle_ma"):
        dec_per_s = np.mean(loops) / (C_BLOCK_S - 1)
        v = bl["mean_v"].mean()
        mj = out["loop_minus_idle_ma"]["mean"] * v / dec_per_s
        out["energy_per_decision_mj"] = float(mj)
        rate = C.FS / C.WIN
        ma = mj * rate / v
        out["projected_controller_ma_at_decision_rate"] = float(ma)
        out["projected_battery_%_per_8h"] = float(100 * ma * 8 / capacity_mah)
    if out.get("live_minus_idle_ma"):
        out["live_battery_%_per_8h"] = float(100 * out["live_minus_idle_ma"]["mean"] * 8 / capacity_mah)
    return out


C_BLOCK_S = 600


def main():
    run_dir = sys.argv[1]
    npz = sys.argv[2] if len(sys.argv) > 2 else os.path.join(C.OUT, "model_fp32.npz")
    res = analyse(run_dir, npz)
    res.to_csv(os.path.join(run_dir, "live_fidelity.csv"), index=False)
    for _, row in res.iterrows():
        for k, v in row.items():
            print(f"{k:28s} {v}")
        print()
    if "--capacity" in sys.argv:
        e = energy(run_dir, float(sys.argv[sys.argv.index("--capacity") + 1]))
        with open(os.path.join(run_dir, "energy.json"), "w") as f:
            json.dump(e, f, indent=2, default=float)
        print(json.dumps({k: v for k, v in e.items() if k != "blocks"}, indent=2, default=float))


if __name__ == "__main__":
    main()
