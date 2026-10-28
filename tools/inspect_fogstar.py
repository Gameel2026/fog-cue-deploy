import sys
import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else r"D:\New research fog work\FoG STAR\sensor_data.csv"
d = pd.read_csv(path)
print("rows", len(d))
print("columns", list(d.columns))
print(d.head(3).to_string())
print()
for c in ["subjectID", "sessionID", "taskID", "activity", "fog"]:
    if c in d:
        print(c, d[c].value_counts(dropna=False).sort_index().to_dict())
print()
keys = [k for k in ["subjectID", "sessionID", "taskID"] if k in d]
dt = d.groupby(keys)["timestamp"].diff()
print("timestamp step (ms) quantiles", dt.quantile([0, 0.01, 0.5, 0.99, 1]).round(3).to_dict())
print("steps > 1.5x median:", int((dt > 1.5 * dt.median()).sum()), " non-positive:", int((dt <= 0).sum()))
same = d.groupby(["subjectID", "sessionID"])["taskID"].apply(lambda s: int((s.diff().fillna(0) != 0).sum()))
print("task changes per subject/session (first 5)", same.head().to_dict())
print()
back = [c for c in d.columns if c.startswith("back_acc")]
for act, name in [(1, "walk"), (3, "stand"), (2, "sit")]:
    g = d[d["activity"] == act][back]
    print(name, "n", len(g))
    print("  mean", g.mean().round(4).to_dict())
    print("  std ", g.std().round(4).to_dict())
