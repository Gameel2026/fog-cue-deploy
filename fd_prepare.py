import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from math import gcd
from scipy.signal import resample_poly

import fd_config as C

KAGGLE_FS = {"tdcsfog": 128, "defog": 100}
FOGSTAR_FS = 60
DAPHNET_FS = 64
DAPHNET_COLS = ["t_ms", "ank_x", "ank_y", "ank_z", "thi_x", "thi_y", "thi_z", "tr_x", "tr_y", "tr_z", "label"]
FOG_COLS = ["StartHesitation", "Turn", "Walking", "Event"]


DROPPED = {"non_finite": 0}


def finite_parts(acc, lab, min_len, extra=None):
    ok = np.isfinite(acc).all(axis=1)
    DROPPED["non_finite"] += int(np.sum(~ok))
    e = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    for a, b in zip(e[::2], e[1::2]):
        if b - a >= min_len:
            yield (acc[a:b], lab[a:b]) if extra is None else (acc[a:b], lab[a:b], extra[a:b])


def to_fs(acc, lab, fs_in):
    if fs_in == C.FS:
        return acc.astype(float), lab.astype(np.int8)
    g = gcd(int(C.FS), int(fs_in)); up, down = C.FS // g, fs_in // g
    x = resample_poly(acc, up, down, axis=0)
    idx = np.minimum(np.round(np.arange(len(x)) * fs_in / C.FS).astype(int), len(lab) - 1)
    return x, lab[idx].astype(np.int8)


def kaggle(name):
    meta = pd.read_csv(os.path.join(C.KAGGLE_ROOT, f"{name}_metadata.csv")).set_index("Id")
    folders = ["tdcsfog"] if name == "tdcsfog" else ["defog", "notype"]
    out = []
    if not os.path.isdir(os.path.join(C.KAGGLE_ROOT, "train")):
        raise FileNotFoundError(f"no train folder in {C.KAGGLE_ROOT}")
    for folder in folders:
        for p in sorted(glob.glob(os.path.join(C.KAGGLE_ROOT, "train", folder, "*.csv"))):
            rid = os.path.splitext(os.path.basename(p))[0]
            d = pd.read_csv(p)
            acc = d[["AccV", "AccML", "AccAP"]].to_numpy(float)
            if name == "tdcsfog":
                acc = acc / C.G
            fog = d[[c for c in FOG_COLS if c in d]].to_numpy().any(axis=1)
            valid = np.ones(len(d), bool)
            for c in ("Valid", "Task"):
                if c in d:
                    valid &= d[c].astype(str).str.lower().eq("true").to_numpy()
            lab = np.where(valid, 1 + fog.astype(int), 0)
            for k, (a, l) in enumerate(finite_parts(acc, lab, KAGGLE_FS[name])):
                x, y = to_fs(a, l, KAGGLE_FS[name])
                out.append({"subject": str(meta.loc[rid, "Subject"]), "recording": f"{folder}/{rid}/{k}", "acc": x, "label": y})
    return out


def fogstar():
    cols = [c for c, _ in C.FOGSTAR_AXES]
    d = pd.read_csv(C.FOGSTAR_CSV)
    d = d.sort_values(["subjectID", "sessionID", "timestamp"], kind="stable").reset_index(drop=True)
    step = 1.0 / FOGSTAR_FS
    out = []
    for (s, sess), g in d.groupby(["subjectID", "sessionID"], sort=True):
        t = g["timestamp"].to_numpy(float)
        brk = np.r_[0, np.flatnonzero((np.diff(t) > 1.5 * step) | (np.diff(t) <= 0)
                                      | (np.diff(g["taskID"].to_numpy()) != 0)) + 1, len(g)]
        for k, (a, b) in enumerate(zip(brk[:-1], brk[1:])):
            seg = g.iloc[a:b]
            acc = np.column_stack([seg[c].to_numpy(float) * sign for c, sign in C.FOGSTAR_AXES])
            lab = 1 + (seg["fog"].to_numpy() == 1).astype(int)
            sev = np.nan_to_num(seg["fog_severity"].to_numpy(float), nan=0.0).astype(int) * (lab == 2)
            for j, (a, l, v) in enumerate(finite_parts(acc, lab, FOGSTAR_FS, sev)):
                x, y = to_fs(a, l, FOGSTAR_FS)
                _, sv = to_fs(a, v, FOGSTAR_FS)
                out.append({"subject": str(int(s)), "recording": f"s{int(sess)}_part{k}_task{int(seg['taskID'].iloc[0])}_{j}",
                            "acc": x, "label": y, "severity": sv})
    return out


def daphnet():
    out = []
    paths = sorted(glob.glob(os.path.join(C.DAPHNET_ROOT, "S*R*.txt")))
    if not paths:
        raise FileNotFoundError(f"no S*R*.txt files in {C.DAPHNET_ROOT}")
    for path in paths:
        b = os.path.splitext(os.path.basename(path))[0]
        d = pd.read_csv(path, sep=r"\s+", header=None, names=DAPHNET_COLS)
        acc = np.column_stack([d[c].to_numpy(float) * sign / 1000.0 for c, sign in C.DAPHNET_AXES])
        for k, (a, l) in enumerate(finite_parts(acc, d["label"].to_numpy().astype(int), DAPHNET_FS)):
            x, y = to_fs(a, l, DAPHNET_FS)
            out.append({"subject": b[1:3], "recording": f"{b}/{k}", "acc": x, "label": y})
    return out


MULTIMODAL_FS = 500


def multimodal():
    paths = sorted(glob.glob(os.path.join(C.MULTIMODAL_ROOT, "**", "task_*.txt"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no task_*.txt files under {C.MULTIMODAL_ROOT}")
    out, skipped = [], []
    cols = [c for c, _ in C.MULTIMODAL_AXES]
    for path in paths:
        rel = os.path.relpath(path, C.MULTIMODAL_ROOT).replace("\\", "/")
        subj = rel.split("/")[0]
        d = pd.read_csv(path, header=None, usecols=cols + [60])
        acc = np.column_stack([d[c].to_numpy(float) * sign / C.MULTIMODAL_LSB_PER_G for c, sign in C.MULTIMODAL_AXES])
        if not np.any(acc != 0):
            skipped.append(rel); continue
        acc[(acc == 0).all(axis=1)] = np.nan
        lab = 1 + (d[60].to_numpy() == 1).astype(int)
        for k, (a, l) in enumerate(finite_parts(acc, lab, MULTIMODAL_FS)):
            x, y = to_fs(a, l, MULTIMODAL_FS)
            out.append({"subject": subj, "recording": f"{os.path.splitext(rel)[0]}/{k}", "acc": x, "label": y})
    print(f"Multimodal: {len(skipped)} task files without a lower-back sensor skipped")
    return out


def prepare(name):
    os.makedirs(C.PREP, exist_ok=True)
    DROPPED["non_finite"] = 0
    loaders = {"FoG-STAR": fogstar, "Daphnet": daphnet, "Multimodal": multimodal}
    recs = loaders[name]() if name in loaders else kaggle(name)
    path = os.path.join(C.PREP, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(recs, f, protocol=4)
    n = sum(len(r["acc"]) for r in recs)
    fog = sum(int(np.sum(r["label"] == 2)) for r in recs)
    print(f"{name}: {len(recs)} recordings, {len({r['subject'] for r in recs})} subjects, {n / C.FS / 3600:.2f} h, "
          f"{DROPPED['non_finite']} non-finite input samples dropped, FoG fraction of labelled {fog / max(sum(int(np.sum(r['label'] > 0)) for r in recs), 1):.3f} -> {path}")


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(C.DATASETS)):
        prepare(name)
