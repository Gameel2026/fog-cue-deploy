import os
import sys
import glob
import numpy as np
import pandas as pd

root = sys.argv[1]
cols = ["t_ms", "ank_x", "ank_y", "ank_z", "thi_x", "thi_y", "thi_z", "tr_x", "tr_y", "tr_z", "label"]
files = sorted(glob.glob(os.path.join(root, "S*R*.txt")))
print("files", len(files), [os.path.basename(f) for f in files])
d = pd.concat([pd.read_csv(f, sep=r"\s+", header=None, names=cols).assign(file=os.path.basename(f)) for f in files])
print("rows", len(d), "label counts", d["label"].value_counts().sort_index().to_dict())
step = d.groupby("file")["t_ms"].diff()
print("time step (ms) quantiles", step.quantile([0, 0.5, 1]).to_dict())
for lab in (1, 2):
    g = d[d["label"] == lab][["tr_x", "tr_y", "tr_z"]]
    print("label", lab, "mean", g.mean().round(1).to_dict(), "std", g.std().round(1).to_dict())
