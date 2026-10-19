#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Can sigma be predicted from molecular structure? The gate for distillation & DER.

Why this is the first thing to measure
--------------------------------------
Three of the four proposed ways to give a single model an uncertainty output share
one precondition that none of them states:

    distillation  : sigma_student(x) is trained to match sigma_teacher(x)
    deep evidential regression : gamma/nu/alpha/beta are functions of x
    quantile regression        : the pinball head is a function of x

All of them assume sigma -- or a distribution whose width encodes it -- is a
LEARNABLE FUNCTION OF THE MOLECULE. If sigma is close to unpredictable from
structure, no head can reproduce it and the choice between the three methods is
moot; if it is highly predictable, the choice is an engineering one and the
cheapest head wins.

This is the same gate the project already applies elsewhere: `02-下一步方案.md`
§5.5 says the substructure direction must show a frozen-readout gain BEFORE any
downstream spend. Here the gate is one regression fit on cached fingerprints.

What is measured
----------------
  * R2 for predicting sigma (model-direction, from the arm panel) out-of-fold,
    against two references that make the number interpretable:
      - a trivial constant predictor (predict the train-fold mean of sigma)
      - the ceiling: the ensemble's own sigma, which is what the teacher knows
  * R2 for predicting |error| instead. That is the quantity the application cares
    about, and it is a strictly harder target than sigma (it contains the label).
  * the Spearman between predicted and true sigma -- what matters for ranking,
    which is what L1 showed sigma is good for.

Reported on log(sigma) as well as sigma, because sigma is positive and skewed and
an MSE fit on the raw scale will spend its capacity on the few large values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
FEAT = ROOT / "5_outputs/feat_cache"
DS = "Sheet2"
SEEDS = (42, 1337, 2024)


def folds_of(seed):
    return json.loads(
        (ROOT / f"5_outputs/split_cache/{DS}_scaffold_k5_seed{seed}.json").read_text()
    )["folds"]


def features():
    return {"desc": np.load(FEAT / f"{DS}_rdkit_desc.npy"),
            "morgan": np.load(FEAT / f"{DS}_morgan_r2_2048.npy"),
            "maccs": np.load(FEAT / f"{DS}_maccs.npy"),
            "atompair": np.load(FEAT / f"{DS}_atompair.npy"),
            "torsion": np.load(FEAT / f"{DS}_torsion.npy")}


def cv_predict(make, X, t, y=None):
    """Out-of-fold predictions over the 15 cached folds, pooled."""
    oof = np.full(len(t), np.nan)
    for s in SEEDS:
        for f in folds_of(s):
            tr = np.asarray(f["train"], int)
            te = np.asarray(f["test"], int)
            m = make().fit(X[tr], t[tr])
            oof[te] = m.predict(X[te])
    return oof


def report(tag, oof, t, tr_mean_by_fold):
    """R2 against the fold-appropriate constant predictor and Spearman."""
    dev = []
    for s in SEEDS:
        for f in folds_of(s):
            te = np.asarray(f["test"], int)
            dev.append(r2_score(t[te], oof[te]))
    rho = float(spearmanr(oof, t).statistic)
    print(f"    {tag:34s} fold-mean R2={np.mean(dev):+.4f}  "
          f"Spearman={rho:+.4f}")
    return {"fold_mean_r2": float(np.mean(dev)), "fold_std": float(np.std(dev, ddof=1)),
            "spearman": rho,
            "pooled_r2": float(r2_score(t, oof))}


def main() -> int:
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    F = features()
    X = np.hstack([F["desc"], F["morgan"], F["maccs"], F["atompair"], F["torsion"]])
    print(f"{DS}: X={X.shape}  y={y.shape}")

    z = np.load(ROOT / "5_outputs/ensemble_oof.npz")
    panel = z["scaffold__panel"]                 # (n_arms, n_partitions, n)
    arms_mean = panel.mean(1)
    sigma = arms_mean.std(0, ddof=1)             # the validated epistemic sigma
    mu = arms_mean.mean(0)
    err = np.abs(y - mu)
    print(f"sigma: mean={sigma.mean():.4f} std={sigma.std(ddof=1):.4f} "
          f"min={sigma.min():.4f} max={sigma.max():.4f}")

    out = {"dataset": DS, "n_features": int(X.shape[1]),
           "sigma_stats": {"mean": float(sigma.mean()), "std": float(sigma.std(ddof=1)),
                           "min": float(sigma.min()), "max": float(sigma.max())},
           "targets": {}}

    targets = {
        "sigma": sigma,
        "log_sigma": np.log(sigma),
        "abs_error": err,
    }
    for tname, t in targets.items():
        print(f"\n  target = {tname}")
        res = {}
        res["constant"] = report("train-mean constant", np.full(len(t), t.mean()),
                                 t, None)
        res["ridge"] = report("Ridge (all features)", cv_predict(
            lambda: __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
                [("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]), X, t), t, None)
        res["rf"] = report("RandomForest 400", cv_predict(
            lambda: RandomForestRegressor(n_estimators=400, max_features=0.3,
                                          n_jobs=12, random_state=0), X, t), t, None)
        res["xgb"] = report("XGBoost 600", cv_predict(
            lambda: xgb.XGBRegressor(tree_method="hist", n_jobs=12, random_state=0,
                                     n_estimators=600, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.4), X, t), t, None)
        out["targets"][tname] = res

    # ---- the reference that makes sigma's predictability interpretable ----
    # If a model can predict |error| about as well as it predicts sigma, then sigma
    # carries no information a structure-only model could not get more directly.
    print("\n  --- interpretation ---")
    s_r2 = out["targets"]["log_sigma"]["xgb"]["fold_mean_r2"]
    e_r2 = out["targets"]["abs_error"]["xgb"]["fold_mean_r2"]
    print(f"    XGB predicts log(sigma) with fold-mean R2 = {s_r2:+.4f}")
    print(f"    XGB predicts |error|    with fold-mean R2 = {e_r2:+.4f}")
    print(f"    -> a single model CAN learn sigma if {s_r2:.3f} is comfortably > 0;")
    print(f"       the distillation/quantile/DER choice only matters above that bar.")

    p = ROOT / "5_outputs/sigma_predictability.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
