import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import fd_config as C
import fd_common as U

if "--quick" in sys.argv:
    U.quick_mode()

import fd_model as M


def folds(pool):
    return list(GroupKFold(n_splits=C.N_FOLDS).split(pool.F, groups=pool.groups))


def validation_split(pool, tr, k):
    g = np.unique(pool.groups[tr])
    rng = np.random.default_rng(C.SEED + k)
    val_g = rng.choice(g, size=max(1, int(round(C.VAL_FRACTION * len(g)))), replace=False)
    m = np.isin(pool.groups[tr], val_g)
    return tr[~m], tr[m]


def cross_validate(pool, name, n_past, cue_factor, splits):
    P = pool.P(n_past)
    oof = np.full(len(pool.F), np.nan); epochs = []
    os.makedirs(os.path.join(C.OUT, "cv"), exist_ok=True)
    for k, (tr, te) in enumerate(splits):
        ck = os.path.join(C.OUT, "cv", f"{name}_fold{k}.npz")
        if os.path.exists(ck):
            d = np.load(ck)
            if np.array_equal(d["te"], te):
                oof[te] = d["p"]; epochs.append(int(d["epoch"])); continue
        fit, val = validation_split(pool, tr, k)
        mu, sd = M.standardisation(pool.F, fit)
        Zs = M.standardised(pool.F, mu, sd)
        t0 = time.time()
        model, ep = M.train(Zs, P, pool.y, fit, C.MAX_EPOCHS, cue_factor, seed=C.SEED + k, val_idx=val)
        p = M.prob(model, Zs, P, te)
        np.savez(ck, te=te, p=p, epoch=ep)
        oof[te] = p; epochs.append(ep)
        print(f"  {name} fold {k + 1}/{len(splits)}: best epoch {ep}, {time.time() - t0:.0f} s", flush=True)
    return oof, epochs


def main():
    os.makedirs(C.OUT, exist_ok=True)
    pool = U.Pool(C.DEVELOPMENT)
    for n in pool.names:
        s = pool.slices[n]
        print(f"{n}: {s.stop - s.start} windows, {len(np.unique(pool.groups[s]))} patients, "
              f"{pool.evaluator(n).hours:.2f} h, {len(pool.evaluator(n).episodes)} FoG episodes", flush=True)
    splits = folds(pool)
    fold = np.zeros(len(pool.F), int)
    for k, (_, te) in enumerate(splits):
        fold[te] = k
    pd.DataFrame({"group": pool.groups, "fold": fold}).drop_duplicates().to_csv(os.path.join(C.OUT, "folds.csv"), index=False)

    rows, oofs, eps = [], {}, {}
    rank = {"primary": 0, "fallback_min_fa": 1, "fallback_max_specificity": 2}
    keys = ["false_alarms_per_hour", "coverage_%", "timely_%", "cue_specificity", "cue_sensitivity", "cue_on_%"]
    for name, (n_past, cue_factor) in C.CANDIDATES.items():
        print(f"candidate {name}: context {n_past} windows, cue weight x{cue_factor}", flush=True)
        oof, epochs = cross_validate(pool, name, n_past, cue_factor, splits)
        oofs[name], eps[name] = oof, epochs
        np.save(os.path.join(C.OUT, "cv", f"{name}_oof.npy"), oof)
        row = {"candidate": name, "n_past": n_past, "cue_factor": cue_factor, "median_best_epoch": float(np.median(epochs))}
        for ctx, ds in C.CONTEXTS.items():
            s = pool.slices[ds]
            table = U.grid(pool.evaluator(ds), oof[s])
            table.to_csv(os.path.join(C.OUT, f"cv_grid_{name}_{ds}.csv"), index=False)
            best, rule = U.operating_point(table)
            row[f"{ctx}_rule"] = rule; row[f"{ctx}_theta_on"] = best["theta_on"]; row[f"{ctx}_theta_off"] = best["theta_off"]
            row[f"{ds}_auc"] = roc_auc_score(pool.y[s], oof[s])
            for k in keys:
                row[f"{ds}_{k}"] = best[k]
        row["worst_rule"] = max(rank[row[f"{c}_rule"]] for c in C.CONTEXTS)
        row["worst_coverage_%"] = min(row[f"{ds}_coverage_%"] for ds in C.CONTEXTS.values())
        row["mean_timely_%"] = float(np.mean([row[f"{ds}_timely_%"] for ds in C.CONTEXTS.values()]))
        row["mean_false_alarms_per_hour"] = float(np.mean([row[f"{ds}_false_alarms_per_hour"] for ds in C.CONTEXTS.values()]))
        rows.append(row)
    cv = pd.DataFrame(rows).sort_values(["worst_rule", "worst_coverage_%", "mean_timely_%", "mean_false_alarms_per_hour"],
                                        ascending=[True, False, False, True], kind="stable")
    cv.to_csv(os.path.join(C.OUT, "cv_candidates.csv"), index=False)
    print(cv.to_string(index=False), flush=True)

    win = cv.iloc[0]; name = win["candidate"]; n_past = int(win["n_past"]); cue_factor = float(win["cue_factor"])
    thetas = {ctx: (float(win[f"{ctx}_theta_on"]), float(win[f"{ctx}_theta_off"])) for ctx in C.CONTEXTS}
    report = {}
    for n in pool.names:
        s = pool.slices[n]; ev = pool.evaluator(n); th = thetas[C.CONTEXT_OF[n]]; on = ev.on(oofs[name][s], th)
        report[n] = {**ev.evaluate(on), **ev.surrogate(on, n_shifts=C.N_SHIFTS), "auc": roc_auc_score(pool.y[s], oofs[name][s]),
                     "context": C.CONTEXT_OF[n], "theta_on": th[0], "theta_off": th[1]}
    pd.DataFrame(report).T.to_csv(os.path.join(C.OUT, "cv_selected.csv"))

    epochs = max(1, int(np.floor(np.median(eps[name]) + 0.5)))
    print(f"selected {name}; final model trained on all {len(np.unique(pool.groups))} patients for {epochs} epochs", flush=True)
    idx = np.arange(len(pool.F))
    mu, sd = M.standardisation(pool.F, idx)
    Zs = M.standardised(pool.F, mu, sd)
    model, _ = M.train(Zs, pool.P(n_past), pool.y, idx, epochs, cue_factor, seed=C.SEED)
    files = {}
    for ctx in C.CONTEXTS:
        prefix = U.model_prefix(ctx)
        M.export(model, mu, sd, thetas[ctx], n_past, prefix)
        files.update({os.path.basename(prefix + s): U.sha256(prefix + s) for s in ["_fp32.npz", "_int8.npz", "_state.pt"]})
    U.write_json({"candidate": name, "n_past": n_past, "cue_factor": cue_factor, "epochs": epochs,
                  "thetas": {k: list(v) for k, v in thetas.items()}, "rules": {c: win[f"{c}_rule"] for c in C.CONTEXTS},
                  "cv": report, "candidates": cv.to_dict(orient="records"), "files": files,
                  "config": U.config_snapshot(), "versions": U.versions(),
                  "created": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(C.OUT, "frozen.json"))
    print("frozen model written to", os.path.join(C.OUT, "frozen.json"), flush=True)


if __name__ == "__main__":
    main()
