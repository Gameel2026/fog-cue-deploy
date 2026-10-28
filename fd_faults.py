import os
import sys
import numpy as np
import pandas as pd
from scipy import signal

import fd_config as C
import fd_common as U

if "--quick" in sys.argv:
    U.quick_mode()

import fd_model as M
from fd_data import stretches, filter_design
from fd_core import features_batch, quantise_rows

DEVICE_B_HZ = 121.56


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def windows(x, win):
    n = len(x) // win
    return x[:n * win].reshape(n, win, x.shape[1])


def causal(acc, sos, zi):
    return signal.sosfilt(sos, acc, axis=0, zi=zi[:, :, None] * acc[0][None, None, :])[0]


def at_rate(acc, fs_out):
    t_in = np.arange(len(acc)) / C.FS
    t_out = np.arange(int(np.floor(t_in[-1] * fs_out)) + 1) / fs_out
    return np.column_stack([np.interp(t_out, t_in, acc[:, a]) for a in range(acc.shape[1])])


def fault_windows(fault, acc, sos, zi):
    if fault == "reference":
        return windows(causal(acc, sos, zi), C.WIN), None
    if fault == "filter_reset_each_window":
        return np.stack([causal(w, sos, zi) for w in windows(acc, C.WIN)]), None
    if fault == "filter_zero_initial_state":
        return windows(signal.sosfilt(sos, acc, axis=0), C.WIN), None
    if fault == "zero_phase_filter":
        return windows(signal.sosfiltfilt(sos, acc, axis=0), C.WIN), None
    if fault == "units_m_s2":
        return windows(causal(acc * C.G, sos, zi), C.WIN), None
    if fault == "axes_ml_ap_swapped":
        return windows(causal(acc[:, [0, 2, 1]], sos, zi), C.WIN), None
    if fault == "device_b_rate_not_resampled":
        x = at_rate(acc, DEVICE_B_HZ)
        return windows(causal(x, sos, zi), C.WIN), DEVICE_B_HZ
    if fault == "device_b_rate_resampled":
        t_b = np.arange(int(np.floor((len(acc) - 1) / C.FS * DEVICE_B_HZ)) + 1) / DEVICE_B_HZ
        xb = np.column_stack([np.interp(t_b, np.arange(len(acc)) / C.FS, acc[:, a]) for a in range(3)])
        t_r = np.arange(len(acc)) / C.FS
        t_r = t_r[t_r <= t_b[-1]]
        xr = np.column_stack([np.interp(t_r, t_b, xb[:, a]) for a in range(3)])
        return windows(causal(xr, sos, zi), C.WIN), None
    raise ValueError(fault)


FAULTS = ["reference", "filter_reset_each_window", "filter_zero_initial_state", "zero_phase_filter", "units_m_s2",
          "axes_ml_ap_swapped", "device_b_rate_not_resampled", "device_b_rate_resampled",
          "features_float32", "context_reset_each_window", "threshold_without_hysteresis", "weights_int4"]


def past_rows(n, n_past):
    j = np.arange(n)[:, None] - (n_past - 1) + np.arange(n_past)[None, :]
    return np.where(j >= 0, j, -1)


def run_stretch(model, mu, sd, theta, n_past, fault, acc, sos, zi, prob_fn):
    feat_fault = "reference" if fault in ("features_float32", "context_reset_each_window", "threshold_without_hysteresis", "weights_int4") else fault
    W, fs_dev = fault_windows(feat_fault, acc, sos, zi)
    dtype = np.float32 if fault == "features_float32" else np.float64
    F = features_batch(W, C.FS, C.NFFT, C.MODE_RES, dtype)
    Z = M.standardised(F, mu, sd)
    P = past_rows(len(Z), 1 if fault == "context_reset_each_window" else n_past)
    if fault == "context_reset_each_window":
        P = np.hstack([np.full((len(Z), n_past - 1), -1), P])
    p = prob_fn(Z, P)
    if fault == "threshold_without_hysteresis":
        on = p >= theta[0]
    else:
        on = hysteresis(p, theta)
    if fs_dev is None:
        return p, on
    n_ref = len(acc) // C.WIN
    t_end_dev = (np.arange(len(p)) + 1) * C.WIN / fs_dev
    t_end_ref = (np.arange(n_ref) + 1) * C.WIN / C.FS
    k = np.searchsorted(t_end_dev, t_end_ref + 1e-12, side="right") - 1
    p_ref_grid = np.where(k >= 0, p[np.maximum(k, 0)], 0.0)
    on_ref_grid = np.where(k >= 0, on[np.maximum(k, 0)], False)
    return p_ref_grid, on_ref_grid


def hysteresis(p, theta):
    out = np.zeros(len(p), bool); st = False
    for t, v in enumerate(p):
        st = (v >= theta[1]) if st else (v >= theta[0]); out[t] = st
    return out


def main():
    names = arg("--datasets", ",".join([C.SELECTION_DATASET] + C.EXTERNAL)).split(",")
    sos, zi = filter_design()
    rows = []
    for name in names:
        prefix = U.prefix_for(name)
        model, mu, sd, theta, n_past = M.load(prefix)
        pool = U.Pool([name]); ev = pool.evaluator(name); meta = pool.metas[name]
        res = {}
        for fault in FAULTS:
            if fault == "weights_int4":
                from fd_precision import weights, batch_prob
                layers, Wh, bh = weights(prefix + "_fp32.npz", 4)
                fn = lambda Z, P: batch_prob(layers, Wh, bh, Z, P, None)
            else:
                fn = lambda Z, P: _prob(model, Z, P)
            ps, ons = [], []
            for _, _, _, acc, _ in stretches(name):
                p, on = run_stretch(model, mu, sd, theta, n_past, fault, acc, sos, zi, fn)
                n = len(acc) // C.WIN
                if len(p) < n:
                    p = np.r_[p, np.repeat(p[-1:], n - len(p))]; on = np.r_[on, np.repeat(on[-1:], n - len(on))]
                ps.append(p[:n]); ons.append(on[:n])
            res[fault] = (np.concatenate(ps), np.concatenate(ons))
            if len(res[fault][0]) != len(meta):
                raise RuntimeError(f"{name} {fault}: {len(res[fault][0])} windows vs {len(meta)}")
        p0, on0 = res["reference"]; m0 = ev.evaluate(on0)
        for fault in FAULTS:
            p, on = res[fault]; m = ev.evaluate(on)
            pa = [100 * float(np.mean(on[g.index] == on0[g.index])) for _, g in meta.groupby("subject")]
            rows.append({"dataset": name, "fault": fault, "decision_agreement_%": 100 * float(np.mean(on == on0)),
                         "min_patient_agreement_%": float(np.min(pa)), "max_abs_dp": float(np.max(np.abs(p - p0))),
                         "false_alarms_per_hour": m["false_alarms_per_hour"], "coverage_%": m["coverage_%"], "timely_%": m["timely_%"],
                         "delta_false_alarms_per_hour": m["false_alarms_per_hour"] - m0["false_alarms_per_hour"],
                         "delta_coverage_%": m["coverage_%"] - m0["coverage_%"], "delta_timely_%": m["timely_%"] - m0["timely_%"]})
            print(f"{name:9s} {fault:32s} agreement {rows[-1]['decision_agreement_%']:7.3f} %  "
                  f"dFA/h {rows[-1]['delta_false_alarms_per_hour']:+8.2f}  dCov {rows[-1]['delta_coverage_%']:+7.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "fault_injection.csv"), index=False)


def _prob(model, Z, P):
    return M.prob(model, Z, P, np.arange(len(P)))


if __name__ == "__main__":
    main()
