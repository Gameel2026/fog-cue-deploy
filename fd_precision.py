import os
import sys
import time
import numpy as np
import pandas as pd

import fd_config as C
import fd_common as U

if "--quick" in sys.argv:
    U.quick_mode()

import fd_model as M
from fd_data import stretches, gather
from fd_core import EdgeLSTM, CueController, lstm_batch, quantise_rows
from fd_metrics import Evaluator

VARIANTS = {"fp32": (None, None), "fp16": ("fp16", None), "int8": (8, None), "int6": (6, None), "int4": (4, None), "w8a8": (8, 8)}


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def weights(npz, mode):
    d = np.load(npz)
    def w(k):
        a = d[k].astype(np.float32)
        if mode == "fp16":
            return a.astype(np.float16).astype(np.float32)
        if isinstance(mode, int):
            return quantise_rows(a, mode)
        return a
    layers = [(w(f"weight_ih_l{l}"), w(f"weight_hh_l{l}"), (d[f"bias_ih_l{l}"] + d[f"bias_hh_l{l}"]).astype(np.float32)) for l in range(2)]
    return layers, w("head_weight"), d["head_bias"].astype(np.float32)


def batch_prob(layers, Wh, bh, Zs, P, act_bits):
    out = []
    for k in range(0, len(P), 4096):
        out.append(lstm_batch(layers, Wh, bh, gather(Zs, P[k:k + 4096]), act_bits))
    return np.concatenate(out)


def payload_bytes(npz, mode):
    d = np.load(npz); total = 0
    for k in d.files:
        if not (k.startswith("weight_") or k.startswith("bias_") or k.startswith("head_")) or k.endswith("__scale"):
            continue
        n = d[k].size
        if "weight" in k:
            if mode is None:
                total += 4 * n
            elif mode == "fp16":
                total += 2 * n
            else:
                total += int(np.ceil(n * mode / 8)) + 4 * d[k].shape[0]
        else:
            total += 4 * n
    return total


def operation_count(n_past, n_in=72, H=64):
    lstm = n_past * (4 * H * (n_in + H) + 4 * H * (H + H)) + 2 * H
    nb = C.NFFT // 2 + 1
    return {"lstm_macs": lstm, "dft_macs": 3 * nb * C.WIN * 2, "filter_macs": 3 * C.WIN * 2 * C.FILT_ORDER * 5 // 2,
            "total_macs": lstm + 3 * nb * C.WIN * 2 + 3 * C.WIN * 2 * C.FILT_ORDER * 5 // 2}


def agreement_by_patient(meta, a, b):
    out = []
    for s, g in meta.groupby("subject", sort=True):
        i = g.index.to_numpy(); out.append(100 * float(np.mean(a[i] == b[i])))
    return out


def stream(name, npz, limit):
    model = EdgeLSTM(npz); ps, ons, k = [], [], 0
    for _, _, _, acc, _ in stretches(name):
        if limit is not None and k >= limit:
            break
        c = CueController(model)
        for j in range(len(acc) // model.win):
            p, on, _ = c.step(acc[j * model.win:(j + 1) * model.win]); ps.append(p); ons.append(on)
        k += 1
    return np.array(ps), np.array(ons, bool)


def main():
    names = arg("--datasets", ",".join(C.DEVELOPMENT + C.EXTERNAL)).split(",")
    limit = int(arg("--max-stretches")) if arg("--max-stretches") else None
    fid, var = [], []
    for name in names:
        prefix = U.prefix_for(name); npz = prefix + "_fp32.npz"
        model, mu, sd, theta, n_past = M.load(prefix)
        pool = U.Pool([name]); meta = pool.metas[name]
        Zs = M.standardised(pool.F, mu, sd); P = pool.P(n_past)
        p_ref = M.prob(model, Zs, P, np.arange(len(pool.F)))
        ev = pool.evaluator(name); on_ref = ev.on(p_ref, theta)
        t0 = time.time()
        p_s, on_s = stream(name, npz, limit)
        n = len(p_s)
        if limit is None and n != len(p_ref):
            raise RuntimeError(f"{name}: streaming produced {n} windows, reference has {len(p_ref)}")
        np.save(os.path.join(C.OUT, f"stream_p_{name}.npy"), p_s)
        sub = meta.iloc[:n].reset_index(drop=True)
        pa = agreement_by_patient(sub, on_ref[:n], on_s)
        fid.append({"dataset": name, "windows": n, "patients": len(pa), "max_abs_dp": float(np.max(np.abs(p_s - p_ref[:n]))),
                    "mean_abs_dp": float(np.mean(np.abs(p_s - p_ref[:n]))),
                    "decision_agreement_%": 100 * float(np.mean(on_s == on_ref[:n])), "disagreements": int(np.sum(on_s != on_ref[:n])),
                    "min_patient_agreement_%": float(np.min(pa)), "stream_seconds": time.time() - t0})
        print(fid[-1], flush=True)
        ref_metrics = ev.evaluate(on_ref)
        for vname, (wmode, abits) in VARIANTS.items():
            layers, Wh, bh = weights(npz, wmode)
            p_v = batch_prob(layers, Wh, bh, Zs, P, abits); on_v = ev.on(p_v, theta)
            r = ev.evaluate(on_v)
            var.append({"dataset": name, "variant": vname, "payload_bytes": payload_bytes(npz, wmode),
                        "max_abs_dp_vs_torch": float(np.max(np.abs(p_v - p_ref))),
                        "decision_agreement_%": 100 * float(np.mean(on_v == on_ref)),
                        "min_patient_agreement_%": float(np.min(agreement_by_patient(meta, on_ref, on_v))),
                        **{f"{k}": r[k] for k in ["false_alarms_per_hour", "coverage_%", "timely_%", "cue_specificity"]},
                        **{f"delta_{k}": r[k] - ref_metrics[k] for k in ["false_alarms_per_hour", "coverage_%", "timely_%"]}})
            print(name, vname, f"agreement {var[-1]['decision_agreement_%']:.3f} %", flush=True)
    pd.DataFrame(fid).to_csv(os.path.join(C.OUT, "fidelity_streaming.csv"), index=False)
    pd.DataFrame(var).to_csv(os.path.join(C.OUT, "precision_variants.csv"), index=False)
    U.write_json(operation_count(int(np.load(U.model_prefix("home") + "_fp32.npz")["n_past"])), os.path.join(C.OUT, "operation_count.json"))


if __name__ == "__main__":
    main()
