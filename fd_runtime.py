"""Standard-runtime check (reviewer request, not in the original plan).

Exports the frozen controller to ONNX and to TensorFlow Lite, runs both
runtimes on the same standardised feature sequences as the reference, and
compares probabilities and hysteresis decisions with the PyTorch reference and
with the streaming implementation.

Run from the repository root:  python fd_runtime.py
Needs: torch, onnx, onnxruntime, tensorflow (pip install onnx onnxruntime tensorflow)
Output: results/runtime_variants.csv, results/runtime/*.onnx, *.tflite
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

import fd_config as C
import fd_common as U
import fd_model as M
from fd_data import gather

OUT_DIR = os.path.join(C.OUT, "runtime")
BATCH = 2048
RESET = "--no-reset" not in sys.argv  # TFLite keeps LSTM state between invoke() calls unless reset


def agreement_by_patient(meta, a, b):
    return [100 * float(np.mean(a[g.index] == b[g.index])) for _, g in meta.groupby("subject", sort=True)]


def export_onnx(model, n_past, n_in, path):
    import copy
    model = copy.deepcopy(model).cpu().eval()
    x = torch.zeros(1, n_past, n_in)
    kw = dict(input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "b"}, "logits": {0: "b"}}, opset_version=17)
    try:
        torch.onnx.export(model, x, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, x, path, **kw)
    return path


def export_tflite(model, n_past, n_in, path):
    import tensorflow as tf
    st = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    inp = tf.keras.Input(shape=(n_past, n_in), batch_size=BATCH)
    l1 = tf.keras.layers.LSTM(64, return_sequences=True)
    l2 = tf.keras.layers.LSTM(64, return_sequences=False)
    head = tf.keras.layers.Dense(2)
    out = head(l2(l1(inp)))
    km = tf.keras.Model(inp, out)
    # PyTorch and Keras use the same gate order (i, f, g/c, o)
    for l, layer in enumerate([l1, l2]):
        layer.set_weights([st[f"lstm.weight_ih_l{l}"].T, st[f"lstm.weight_hh_l{l}"].T,
                           st[f"lstm.bias_ih_l{l}"] + st[f"lstm.bias_hh_l{l}"]])
    head.set_weights([st["head.1.weight"].T, st["head.1.bias"]])
    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    open(path, "wb").write(conv.convert())
    return path


def softmax1(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return (e[:, 1] / e.sum(1)).astype(np.float64)


def batches(Zs, P):
    for k in range(0, len(P), BATCH):
        yield np.ascontiguousarray(gather(Zs, P[k:k + BATCH]), dtype=np.float32)


def run_onnx(path, Zs, P):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return np.concatenate([softmax1(s.run(None, {"x": b})[0]) for b in batches(Zs, P)])


def run_tflite(path, Zs, P):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    out = []
    for b in batches(Zs, P):
        n = len(b)
        if n < BATCH:
            b = np.concatenate([b, np.zeros((BATCH - n,) + b.shape[1:], np.float32)])
        if RESET:
            it.reset_all_variables()
        it.set_tensor(i, b); it.invoke(); out.append(softmax1(it.get_tensor(o)[:n]))
    return np.concatenate(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = C.DEVELOPMENT + C.EXTERNAL
    rows, built = [], {}
    for name in names:
        prefix = U.prefix_for(name)
        model, mu, sd, theta, n_past = M.load(prefix)
        if prefix not in built:
            n_in = len(mu); tag = os.path.basename(prefix)
            built[prefix] = {"onnx": export_onnx(model, n_past, n_in, os.path.join(OUT_DIR, tag + ".onnx")),
                             "tflite": export_tflite(model, n_past, n_in, os.path.join(OUT_DIR, tag + ".tflite"))}
        pool = U.Pool([name]); meta = pool.metas[name]; ev = pool.evaluator(name)
        Zs = M.standardised(pool.F, mu, sd); P = pool.P(n_past)
        p_ref = M.prob(model, Zs, P, np.arange(len(pool.F))); on_ref = ev.on(p_ref, theta)
        sp = os.path.join(C.OUT, f"stream_p_{name}.npy")
        p_str = np.load(sp) if os.path.exists(sp) else None
        for rt in ["onnx", "tflite"]:
            t0 = time.time()
            p = run_onnx(built[prefix][rt], Zs, P) if rt == "onnx" else run_tflite(built[prefix][rt], Zs, P)
            on = ev.on(p, theta)
            r = {"dataset": name, "runtime": rt, "windows": len(p), "patients": meta["subject"].nunique(),
                 "max_abs_dp_vs_torch": float(np.max(np.abs(p - p_ref))),
                 "max_abs_dp_vs_stream": float(np.max(np.abs(p - p_str))) if p_str is not None and len(p_str) == len(p) else np.nan,
                 "decision_agreement_%": 100 * float(np.mean(on == on_ref)), "disagreements": int(np.sum(on != on_ref)),
                 "min_patient_agreement_%": float(np.min(agreement_by_patient(meta, on_ref, on))),
                 "file_bytes": os.path.getsize(built[prefix][rt]), "seconds": time.time() - t0}
            rows.append(r); print(r, flush=True)
    import onnxruntime, tensorflow
    df = pd.DataFrame(rows); df["torch"] = torch.__version__; df["onnxruntime"] = onnxruntime.__version__; df["tensorflow"] = tensorflow.__version__
    df.to_csv(os.path.join(C.OUT, "runtime_variants.csv" if RESET else "runtime_variants_noreset.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
