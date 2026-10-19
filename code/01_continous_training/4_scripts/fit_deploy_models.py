#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit and PERSIST the deployed predictor, so screening does not refit every run.

Why this exists
---------------
The deployed predictor is four models (see prompt/26-09-15/10-附录 §7.1):

    mu    = NNLS blend of neural_2d + traditional + xgb_atompair
    sigma = RF(fingerprints) -> log sigma, times the 1.9 calibration factor

Two of those four were never written to disk. `arm_zoo.py` computes the XGB arms
inside `build_arms` and throws them away, and `sigma_predictability.py` fits the
sigma RF purely to read a cross-validated R2 off it. Every screening run would
therefore refit 3 CPU models from scratch -- and, worse, could silently refit them
with different settings and produce a different predictor than the one that was
validated at 0.7143.

This script fits each component with the EXACT configuration that produced the
validated numbers, and dumps them. `predict_lipids.py` then only has to load and
apply them, which also makes the screening run reproducible from artifacts instead
of from code.

Configurations reproduced (all fixed, none tuned here)
------------------------------------------------------
  traditional   desc + morgan_r3_2048 | xgb, n=1200 depth=8 lr=0.03 sub=0.7
                colsample=0.3 lambda=2.0        -> traditional_baseline_tuned.json
                "desc+morgan_r3_2048|xgb", 15-fold 0.6448
  xgb_atompair  atom pair | xgb, n=600 depth=6 lr=0.05 sub=0.8 colsample=0.4
                lambda=1.0                      -> arm_zoo.py build_arms
  sigma RF      all five fingerprints -> log(sigma), 400 trees, max_features=0.3
                -> sigma_predictability.py (fold-mean R2 +0.309)
  neural        NOT fitted here. The 15 fine-tuned fold checkpoints already exist
                under 5_outputs/finetune_moltrain/ and are loaded by the predictor.

The blend weights are read from ensemble_weighted.json rather than refitted: they
are the validated LOPO weights, and refitting them here would introduce a new
degree of freedom into a pipeline that was scored under a fixed protocol.

Usage:
    python 4_scripts/fit_deploy_models.py            # fit + save
    python 4_scripts/fit_deploy_models.py --verify   # also report train-fit R2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
FEAT = ROOT / "5_outputs/feat_cache"
MODELS = ROOT / "5_outputs/deploy_models"

DS = "Sheet2"
# Fold-0 of the scaffold/seed-42 partition. Its training rows are by construction
# disjoint from that fold's test rows, which is the only hold-out this fitting step
# can honour -- the deployment models are meant to be refit on everything.
# ⚠️ A production deployment that must be comparable to 0.7143 should instead use
# the fold checkpoints; see the report's caveat on this point.
FIT_SEED, FIT_FOLD = 42, 0

TRAD = dict(n_estimators=1200, max_depth=8, learning_rate=0.03, subsample=0.7,
            colsample_bytree=0.3, reg_lambda=2.0)
ATOM = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.4, reg_lambda=1.0)


def _feats(names):
    return np.hstack([np.load(FEAT / f"{DS}_{n}.npy") for n in names])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    print(f"{DS}: n={len(y)}")

    folds = json.loads((ROOT / f"5_outputs/split_cache/{DS}_scaffold_k5_seed"
                        f"{FIT_SEED}.json").read_text())["folds"][FIT_FOLD]
    tr = np.asarray(folds["train"], int)
    te = np.asarray(folds["test"], int)
    print(f"fitting on {len(tr)} rows (scaffold seed{FIT_SEED} fold{FIT_FOLD} "
          f"TRAIN split); {len(te)} rows of that fold stay unseen")

    import xgboost as xgb
    manifest = {"dataset": DS, "target": "RLU (log2)", "n_fit": int(len(tr)),
                "fit_source": f"scaffold_seed{FIT_SEED}_fold{FIT_FOLD}_train",
                "components": {}}

    def fit_xgb(name, X, params):
        t0 = time.time()
        m = xgb.XGBRegressor(tree_method="hist", n_jobs=12, random_state=0, **params)
        m.fit(X[tr], y[tr])
        m.save_model(str(MODELS / f"{name}.json"))
        r2 = float(1 - np.sum((y[te] - m.predict(X[te])) ** 2)
                   / np.sum((y[te] - y[tr].mean()) ** 2)) if args.verify else None
        manifest["components"][name] = {
            "kind": "xgboost", "params": params, "features": list(FEATS[name]),
            "n_features": int(X.shape[1]), "fit_seconds": round(time.time() - t0, 1),
            "holdout_fold_r2": r2}
        print(f"  {name:14s} {X.shape[1]:>5} feats  {time.time()-t0:6.1f}s"
              + (f"   fold-0 R2={r2:+.4f}" if r2 is not None else ""))

    FEATS = {"traditional": ("rdkit_desc", "morgan_r3_2048"),
             "xgb_atompair": ("atompair",)}

    # ---- pin the descriptor column spec BEFORE anything is fitted ----
    # `arm_zoo.rdkit_descriptors` drops constant/oversized columns PER CALL. Sheet2's
    # 1920 molecules keep 132 columns; a 200-molecule screening batch keeps 113. A
    # model fitted on one width and applied to the other does NOT error -- XGBoost
    # aligns by position and returns plausible nonsense. Persist the surviving column
    # NAMES and the imputation medians so new molecules can never move the columns.
    # Verified: replaying this spec on Sheet2 reproduces the cached 132-column matrix
    # bit-for-bit.
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    smis = d["SMILES"].astype(str).str.strip().tolist()
    keys0 = list(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smis[0])).keys())
    raw = np.asarray(
        [[(lambda v: v[k] if v.get(k) is not None else np.nan)(
            Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))) for k in keys0]
         for s in smis], dtype=np.float64)
    raw[~np.isfinite(raw)] = np.nan
    med = np.nanmedian(raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(~np.isfinite(raw))
    raw[idx] = np.take(med, idx[1])
    mask = (raw.std(axis=0) > 1e-8) & (np.abs(raw).max(axis=0) <= 1e6)
    kept = [k for k, m in zip(keys0, mask) if m]
    np.savez(MODELS / "desc_spec.npz", keys=np.array(kept), med=med[mask])
    cached = np.load(FEAT / f"{DS}_rdkit_desc.npy")
    same = (len(kept) == cached.shape[1]
            and np.allclose(raw[:, mask].astype(np.float32), cached, equal_nan=True))
    print(f"  {'desc_spec':14s} {len(keys0)} raw descs -> {len(kept)} kept; "
          f"reproduces the cached matrix: {same}")
    assert same, ("the pinned descriptor spec does not reproduce the cached matrix "
                  "-- refusing to persist a spec that disagrees with what the models "
                  "were validated on")
    manifest["components"]["desc_spec"] = {
        "raw_descriptors": len(keys0), "kept": len(kept),
        "rule": "drop std<=1e-8 or |max|>1e6, computed on Sheet2, then FROZEN",
        "file": "desc_spec.npz", "reproduces_cached_matrix": bool(same)}

    print("\nfitting CPU arms (configs copied verbatim from the validated runs)")
    fit_xgb("traditional", _feats(FEATS["traditional"]), TRAD)
    fit_xgb("xgb_atompair", _feats(FEATS["xgb_atompair"]), ATOM)

    # ---- sigma RF ----
    z = np.load(ROOT / "5_outputs/ensemble_oof.npz")
    sigma = z["scaffold__panel"].mean(1).std(0, ddof=1)
    SIG_FEATS = ("rdkit_desc", "morgan_r2_2048", "maccs", "atompair", "torsion")
    Xs = _feats(SIG_FEATS)
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=400, max_features=0.3, n_jobs=12,
                               random_state=0).fit(Xs[tr], np.log(sigma)[tr])
    joblib.dump(rf, MODELS / "sigma_hat_rf.joblib")
    print(f"  {'sigma_hat':14s} {Xs.shape[1]:>5} feats  {time.time()-t0:6.1f}s"
          f"   (target = log sigma, model-direction spread across the arm panel)")
    manifest["components"]["sigma_hat"] = {
        "kind": "RandomForestRegressor", "n_estimators": 400, "max_features": 0.3,
        "features": list(SIG_FEATS), "n_features": int(Xs.shape[1]),
        "target": "log(sigma), sigma = std over arms of per-arm mean-over-partitions",
        "file": "sigma_hat_rf.joblib"}

    # ---- blend weights: read, never refit ----
    w = json.loads((ROOT / "5_outputs/ensemble_weighted.json").read_text())
    per_protocol = {p: blk["candidates"]["xgb_atompair"]["weights"]
                    for p, blk in w["protocols"].items()}
    # Deployment uses the scaffold weights (15 folds, 3 partitions, fully symmetric).
    blend_w = np.asarray(per_protocol["scaffold"][0], float)   # seed-42 LOPO weights
    manifest["blend"] = {
        "order": ["neural_2d", "traditional", "xgb_atompair"],
        "weights_used": blend_w.tolist(),
        "source": "ensemble_weighted.json -> protocols.scaffold.candidates."
                  "xgb_atompair.weights[0]  (seed-42 leave-one-partition-out)",
        "all_protocol_weights": per_protocol,
        "note": "NOT refit here. Refitting would add a free parameter to a pipeline "
                "scored under a frozen protocol. The three LOPO vectors are within "
                "~0.04 of each other for scaffold, so the choice is not load-bearing."}
    manifest["calibration"] = {
        "sigma_scale_factor": 1.9,
        "source": "P1 report §4.4; |z|<1 was 0.423 vs Gaussian 0.683, inflate to "
                  "make intervals honest. Sorting is unaffected.",
        "apply_to": "sigma, at inference: sigma_hat_calibrated = 1.9 * exp(rf.predict)"}

    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  blend weights (scaffold, seed-42 LOPO): {blend_w}")
    print(f"  calibration factor: x1.9")
    print(f"\nwrote {MODELS}/  (traditional.json, xgb_atompair.json, "
          f"sigma_hat_rf.joblib, manifest.json)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "4_scripts"))
    raise SystemExit(main())
