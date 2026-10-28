import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fd_config as C
from fd_data import stretches, filter_design, relabel, PRE, FOG, POST, NOFOG
from fd_core import features_batch
from scipy import signal

STILL_SD = 0.02


def windows(name):
    sos, zi = filter_design()
    rows = []
    for subj, rec, seg, acc, lab in stretches(name):
        n = len(acc) // C.WIN
        raw = acc[:n * C.WIN].reshape(n, C.WIN, 3)
        xf = signal.sosfilt(sos, acc, axis=0, zi=zi[:, :, None] * acc[0][None, None, :])[0][:n * C.WIN].reshape(n, C.WIN, 3)
        y2 = np.array([2 if np.sum(lab[k * C.WIN:(k + 1) * C.WIN] == 2) >= C.FOG_MIN_SAMPLES else 1 for k in range(n)])
        y4 = relabel(y2, n)
        F = features_batch(xf, C.FS, C.NFFT, C.MODE_RES)
        rows.append(pd.DataFrame({"subject": subj, "y4": y4,
                                  "g_v": raw[:, :, 0].mean(1), "g_ml": raw[:, :, 1].mean(1), "g_ap": raw[:, :, 2].mean(1),
                                  "sd_v": xf[:, :, 0].std(1), "sd_ap": xf[:, :, 2].std(1),
                                  "loco_v": F[:, 14], "freeze_v": F[:, 15], "fi_v": F[:, 16],
                                  "loco_ap": F[:, 24 * 2 + 14], "freeze_ap": F[:, 24 * 2 + 15], "fi_ap": F[:, 24 * 2 + 16],
                                  "peak_v": F[:, 23]}))
    return pd.concat(rows, ignore_index=True)


def auc(a, b):
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    s = np.r_[a, b]
    return roc_auc_score(y, s) if len(a) and len(b) else np.nan


def main():
    out = []
    for name in (sys.argv[1:] or list(C.DATASETS)):
        w = windows(name)
        no, fog, pre = w[w.y4 == NOFOG], w[w.y4 == FOG], w[w.y4 == PRE]
        r = {"dataset": name, "windows": len(w), "fog_%": round(100 * (w.y4 == FOG).mean(), 1)}
        for k in ["g_v", "g_ml", "g_ap"]:
            r[f"{k}_mean_nofog"] = round(no[k].mean(), 3)
        r["still_%_nofog"] = round(100 * (no.sd_v < STILL_SD).mean(), 1)
        r["still_%_pre"] = round(100 * (pre.sd_v < STILL_SD).mean(), 1)
        r["still_%_fog"] = round(100 * (fog.sd_v < STILL_SD).mean(), 1)
        r["sd_v_median_nofog"] = round(no.sd_v.median(), 4)
        r["sd_v_median_fog"] = round(fog.sd_v.median(), 4)
        r["peak_hz_median_nofog"] = round(no.peak_v.median(), 2)
        r["peak_hz_median_fog"] = round(fog.peak_v.median(), 2)
        r["auc_fog_vs_nofog_fi_v"] = round(auc(no.fi_v, fog.fi_v), 3)
        r["auc_fog_vs_nofog_fi_ap"] = round(auc(no.fi_ap, fog.fi_ap), 3)
        r["auc_fog_vs_nofog_freeze_ap"] = round(auc(no.freeze_ap, fog.freeze_ap), 3)
        r["auc_pre_vs_nofog_fi_ap"] = round(auc(no.fi_ap, pre.fi_ap), 3)
        r["auc_pre_vs_nofog_sd_v"] = round(auc(no.sd_v, pre.sd_v), 3)
        r["auc_pre_vs_nofog_loco_v"] = round(auc(no.loco_v, pre.loco_v), 3)
        moving = no[no.sd_v >= STILL_SD]
        r["auc_pre_vs_moving_nofog_fi_ap"] = round(auc(moving.fi_ap, pre.fi_ap), 3)
        out.append(r)
        print(name, "done", flush=True)
    print(pd.DataFrame(out).set_index("dataset").T.to_string())


if __name__ == "__main__":
    main()
