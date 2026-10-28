import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fd_config as C
from fd_data import stretches, runs

rows = []
for name in (sys.argv[1:] or list(C.DATASETS)):
    hours, subj_fog, subj = 0.0, set(), set()
    eps = []
    for s, rec, seg, acc, lab in stretches(name):
        n = len(acc) // C.WIN
        y2 = np.array([2 if np.sum(lab[k * C.WIN:(k + 1) * C.WIN] == 2) >= C.FOG_MIN_SAMPLES else 1 for k in range(n)])
        hours += n * C.WIN / C.FS / 3600; subj.add(s)
        prev = 0
        for a, b in runs(y2 == 2):
            eps.append({"lead_s": (a - prev) * C.WIN / C.FS, "onset_from_start_s": a * C.WIN / C.FS,
                        "dur_s": (b - a) * C.WIN / C.FS, "eligible": (a - prev) >= C.PRE_W, "subject": s})
            prev = b; subj_fog.add(s)
    e = pd.DataFrame(eps)
    if len(e) == 0:
        rows.append({"dataset": name, "hours": hours, "patients": len(subj), "episodes": 0}); continue
    rows.append({"dataset": name, "hours": round(hours, 2), "patients": len(subj), "patients_with_fog": len(subj_fog),
                 "episodes": len(e), "eligible_for_prediction": int(e["eligible"].sum()),
                 "eligible_%": round(100 * e["eligible"].mean(), 1),
                 "episodes_per_hour": round(len(e) / hours, 1),
                 "median_duration_s": round(e["dur_s"].median(), 1),
                 "median_clean_lead_s": round(e["lead_s"].median(), 1),
                 "onset_within_4s_of_start_%": round(100 * (e["onset_from_start_s"] < 4).mean(), 1),
                 "patients_with_eligible_episode": int(e[e["eligible"]]["subject"].nunique())})
print(pd.DataFrame(rows).T.to_string())
