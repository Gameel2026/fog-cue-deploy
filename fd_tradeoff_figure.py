"""Coverage and false-alarm trade-off of the selected candidate in cross-validation (manuscript Fig. 2).

Reads the threshold grids written by fd_develop.py and marks the frozen operating points.
Run from the repository root:  python fd_tradeoff_figure.py
Output: figures/fig_tradeoff.png
"""
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fd_config as C


def main():
    frozen = json.load(open(os.path.join(C.OUT, "frozen.json")))
    cand = frozen["candidate"]
    th = {k: tuple(v) for k, v in frozen["thetas"].items()}
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    sty = {"tdcsfog": ("tDCS-FoG (laboratory)", "#1f4e79", "o", th["lab"]),
           "defog": ("DeFOG (home)", "#c55a11", "s", th["home"])}
    for ds, (lab, c, m, t) in sty.items():
        d = pd.read_csv(os.path.join(C.OUT, f"cv_grid_{cand}_{ds}.csv"))
        ax.scatter(d.false_alarms_per_hour, d["coverage_%"], s=6, color=c, alpha=0.35, marker=m, linewidths=0)
        d = d.sort_values("false_alarms_per_hour")
        best, xs, ys = -1, [], []
        for _, r in d.iterrows():
            if r["coverage_%"] > best:
                best = r["coverage_%"]; xs.append(r.false_alarms_per_hour); ys.append(best)
        ax.plot(xs, ys, color=c, lw=1, label=lab)
        s = d[(d.theta_on == t[0]) & (d.theta_off == t[1])].iloc[0]
        ax.scatter([s.false_alarms_per_hour], [s["coverage_%"]], s=40, color=c, marker=m,
                   edgecolors="black", linewidths=0.8, zorder=5)
    ax.axvline(C.FA_TARGET, color="gray", ls="--", lw=0.8)
    ax.text(C.FA_TARGET + 0.8, 80, f"{C.FA_TARGET:g} FA/h limit", fontsize=7, color="gray")
    ax.set_xlabel("False alarms per hour"); ax.set_ylabel("Episode coverage (%)")
    ax.set_xlim(0, 60); ax.set_ylim(0, 85)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    os.makedirs("figures", exist_ok=True)
    fig.savefig(os.path.join("figures", "fig_tradeoff.png"), dpi=300)
    print("written figures/fig_tradeoff.png")


if __name__ == "__main__":
    main()
