import os
import sys
import struct
import numpy as np

import fd_config as C
import fd_common as U

if "--quick" in sys.argv:
    U.quick_mode()

from fd_data import stretches
from fd_core import EdgeLSTM, CueController


def model_bin(npz, path):
    d = np.load(npz)
    f32 = lambda k: np.ascontiguousarray(d[k], np.float32).tobytes()
    f64 = lambda a: np.ascontiguousarray(a, np.float64).tobytes()
    nf, H = d["weight_ih_l0"].shape[1], d["weight_hh_l0"].shape[1]
    with open(path, "wb") as f:
        f.write(b"FOG2")
        f.write(struct.pack("<6i", nf, H, int(d["n_past"]), int(d["win"]), int(d["nfft"]), d["sos"].shape[0]))
        f.write(struct.pack("<4d", float(d["fs"]), float(d["mode_res"]), float(d["theta_on"]), float(d["theta_off"])))
        f.write(f64(d["mu"]) + f64(d["sd"]) + f64(d["sos"]) + f64(d["zi_unit"]))
        b0 = (d["bias_ih_l0"] + d["bias_hh_l0"]).astype(np.float32).tobytes()
        b1 = (d["bias_ih_l1"] + d["bias_hh_l1"]).astype(np.float32).tobytes()
        f.write(f32("weight_ih_l0") + f32("weight_hh_l0") + b0 + f32("weight_ih_l1") + f32("weight_hh_l1") + b1
                + f32("head_weight") + f32("head_bias"))


def python_stream(model, acc):
    c = CueController(model)
    return np.array([c.step(acc[k * model.win:(k + 1) * model.win])[0] for k in range(len(acc) // model.win)])


def replay_bin(model, name, path):
    acc = max((a for _, _, _, a, _ in stretches(name)), key=len)
    p = python_stream(model, acc)
    with open(path, "wb") as f:
        f.write(b"FOGR" + struct.pack("<2i", len(acc), len(p)))
        f.write(np.ascontiguousarray(acc, np.float64).tobytes() + p.astype(np.float64).tobytes())
    np.save(path.replace(".bin", "_p.npy"), p)
    return len(acc), len(p)


def streams_bin(model, name, path):
    st = [a for _, _, _, a, _ in stretches(name)]
    with open(path, "wb") as f:
        f.write(b"FOGS" + struct.pack("<2i", len(st), model.win))
        for a in st:
            n = len(a) // model.win
            f.write(struct.pack("<i", n) + np.ascontiguousarray(a[:n * model.win], np.float64).tobytes())
    return len(st), sum(len(a) // model.win for a in st)


def main():
    out = os.path.join(C.OUT, "edge"); os.makedirs(out, exist_ok=True)
    for ctx in C.CONTEXTS:
        npz = U.model_prefix(ctx) + "_fp32.npz"
        fn = "model.bin" if ctx == "home" else f"model_{ctx}.bin"
        model_bin(npz, os.path.join(out, fn))
        print(fn, os.path.getsize(os.path.join(out, fn)), "bytes")
    model = EdgeLSTM(U.model_prefix("home") + "_fp32.npz")
    print("replay.bin samples, windows:", replay_bin(model, C.SELECTION_DATASET, os.path.join(out, "replay.bin")))
    with open(os.path.join(out, "streams_models.txt"), "w", newline="\n") as f:
        for name in [C.SELECTION_DATASET] + C.EXTERNAL:
            print(f"streams_{name}.bin stretches, windows:", streams_bin(model, name, os.path.join(out, f"streams_{name}.bin")))
            ctx = C.CONTEXT_OF[name]
            f.write(f"{name} {'model.bin' if ctx == 'home' else f'model_{ctx}.bin'}\n")
    U.write_json({k: U.sha256(os.path.join(out, k)) for k in sorted(os.listdir(out))}, os.path.join(out, "hashes.json"))


if __name__ == "__main__":
    main()
