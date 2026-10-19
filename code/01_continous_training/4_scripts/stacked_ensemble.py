#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A stacker that learns WHERE each arm is reliable, instead of averaging blindly.

Why this exists. The error attribution found a strong, monotone structure in the
neural-vs-traditional difference by label value:

    y in (-5.5, 0.09]   neural  wins   dE = -0.98   (below detection limit)
    y in ( 1.66, 5.88]  traditional wins  dE = +0.80..+1.30
    y in ( 8.27, 16.2]  neural  wins   dE = -5.20   (top transfection)

A plain 50/50 average cannot exploit that: it applies one weight everywhere, so it
must trade off the mid-range (where the tree is better) against the tails (where the
network is far better). A stacked meta-model can assign different weights per region.

Method. Base features are the three arms' out-of-fold predictions. The meta-model is
a small gradient-boosted regressor, refit per fold on the other folds' OOF predictions
and scored on the held-out fold -- so the reported number is genuinely out-of-sample at
the meta level too.

Honest caveat, reported alongside the result: a base molecule's OOF prediction comes
from the model trained on the OTHER folds, so when the stacker trains on folds != k,
some of those rows were seen (as training data) by the fold-k base model. That is the
standard stacking setup, but it means the stacker's advantage is an upper bound, not a
guaranteed deployment number. The comparison that matters is stacker-vs-simple-average
on the SAME folds, since both inherit the same caveat.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
OUT = ROOT / "5_outputs/stacked_ensemble.json"

DS = "Sheet2"
TARGET_COL = "RLU (log2)"
SEEDS = (42, 1337, 2024)
EXP = ("Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed{}"
       "_lr5e-05_ep40")

ARMS = ("neural_2d", "neural_3d", "traditional")


def load_all():
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", TARGET_COL])
    y = pd.to_numeric(d[TARGET_COL]).values.astype(float)
    trad = np.load(ROOT / "5_outputs/traditional_oof.npz")
    folds_by_seed = {}
    preds_by_seed = {}
    for seed in SEEDS:
        p = ROOT / f"5_outputs/split_cache/{DS}_scaffold_k5_seed{seed}.json"
        folds_by_seed[seed] = [(np.asarray(f["train"], int), np.asarray(f["test"], int))
                               for f in json.loads(p.read_text())["folds"]]
        row = {}
        for arm in ("neural_2d", "neural_3d"):
            rp = ROOT / "5_outputs/finetune_moltrain" / (
                EXP.format(seed) + ("_3dconf" if arm == "neural_3d" else ""))
            z = np.asarray(joblib.load(rp / "cv.data")).ravel()
            sc = joblib.load(rp / "target_scaler.ss")
            row[arm] = sc.inverse_transform(z.reshape(-1, 1)).ravel()
        row["traditional"] = trad[f"pred_{seed}"]
        preds_by_seed[seed] = row
    return y, folds_by_seed, preds_by_seed


def main():
    y, folds_by_seed, preds_by_seed = load_all()
    print(f"{DS}: n={len(y)}, arms={ARMS}\n")

    import xgboost as xgb

    rows = []
    for seed in SEEDS:
        P = preds_by_seed[seed]
        folds = folds_by_seed[seed]
        X = np.column_stack([P[a] for a in ARMS])

        stack_pred = np.zeros(len(y))
        avg3_pred = np.zeros(len(y))
        avg2_pred = np.zeros(len(y))
        for tr, te in folds:
            meta = xgb.XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=8, tree_method="hist", random_state=0)
            meta.fit(X[tr], y[tr])
            stack_pred[te] = meta.predict(X[te])
            avg3_pred[te] = X[te].mean(axis=1)
            avg2_pred[te] = X[te][:, [0, 2]].mean(axis=1)   # the best 2-arm pairing

        from sklearn.metrics import r2_score as r2
        for name, pred in (("stack_xgb", stack_pred), ("avg3", avg3_pred),
                           ("avg_neural2d+trad", avg2_pred)):
            pf = [float(r2(y[te], pred[te])) for _tr, te in folds]
            rows.append(dict(seed=seed, arm=name, folds=pf))

        # feature importance: does the meta-model actually use the arms differently?
        meta_full = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                     n_jobs=8, tree_method="hist", random_state=0)
        meta_full.fit(X, y)
        imp = meta_full.feature_importances_
        print(f"seed {seed}: meta importance " +
              "  ".join(f"{a}={v:.3f}" for a, v in zip(ARMS, imp)))

    df = pd.DataFrame(rows)
    print(f"\n{'arm':22s} {'mean R2':>9} {'std':>8} {'n folds':>8}")
    summary = {}
    for arm, grp in df.groupby("arm"):
        allf = np.concatenate([np.asarray(v) for v in grp["folds"]])
        summary[arm] = allf
        print(f"{arm:22s} {allf.mean():>9.4f} {allf.std(ddof=1):>8.4f} {len(allf):>8}")

    base = summary["avg3"]
    print(f"\npaired vs average of all 3 (the current 0.7079 baseline):")
    for arm in ("stack_xgb", "avg_neural2d+trad"):
        if arm not in summary or len(summary[arm]) != len(base):
            continue
        d = summary[arm] - base
        t, p = ttest_rel(summary[arm], base)
        print(f"  {arm:22s} {summary[arm].mean():.4f}  delta {d.mean():+.4f}  "
              f"t={t:5.2f} p={p:.4f}  wins {int((d>0).sum())}/{len(d)}")

    OUT.write_text(json.dumps({
        "arms": ARMS,
        "means": {k: round(float(v.mean()), 4) for k, v in summary.items()},
        "stds": {k: round(float(v.std(ddof=1)), 4) for k, v in summary.items()},
        "per_fold": {k: [round(float(x), 4) for x in v] for k, v in summary.items()},
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
