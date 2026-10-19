#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strong classic baseline (ECFP4 + full RDKit descriptors + XGBoost) on the EXACT
cached scaffold folds the neural fine-tune uses, plus a label-censoring ablation.

Why: fine-tuned LipidFM reports Sheet2 scaffold s1337 pooled/mean R2 ~ 0.67-0.69.
The project's earlier retrospective (prompt/26-08-18/06) measured ECFP4+RF at
Pearson 0.8442 but under the LUMI *sequential* protocol and a different metric, so
the two numbers were never directly comparable. This script puts both on the same
split AND the same metric.

The censoring ablation: 16.1% of Sheet2 labels are <= 0 (log2(raw/empty) at or
below the detection limit). Those rows are censored, not measured — MSE treats the
clipped value as an exact target, so the model is fit partly to an artifact.
Training on the uncensored subset and scoring on all rows tests how much of the
reported ceiling is label censoring rather than chemistry.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import r2_score

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")

SPECS = {
    "Sheet2": ("Sheet2", "RLU (log2)", False),
    # Sheet3's target gained an IGROV1_ prefix in the 2026-09-13 data fix.
    "Sheet3": ("Sheet3", ("IGROV1_transfection_efficiency", "transfection_efficiency"), True),
}


def load(ds):
    sh, col, log10 = SPECS[ds]
    d = pd.read_excel(XLSX, sheet_name=sh)
    if isinstance(col, tuple):
        col = next((c for c in col if c in d.columns), None)
        if col is None:
            raise SystemExit(f"Sheet3 target column not found in {list(d.columns)}")
    d = d.dropna(subset=["SMILES", col])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d[col]).values.astype(float)
    if log10:
        y = np.log10(y)
    return smis, y


def featurize(smis):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    desc_keys = None
    fps, descs = [], []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        fps.append(np.asarray(gen.GetFingerprint(m), dtype=np.uint8))
        v = Descriptors.CalcMolDescriptors(m)
        if desc_keys is None:
            desc_keys = list(v.keys())
        descs.append([v[k] if v[k] is not None else np.nan for k in desc_keys])
    D = np.asarray(descs, dtype=np.float64)
    # RdKit emits inf for a few descriptors (e.g. Ipc) on large flexible lipids.
    # Both inf and nan are unmeasurable -> median-impute, then drop constant cols.
    D[~np.isfinite(D)] = np.nan
    med = np.nanmedian(D, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(~np.isfinite(D))
    D[idx] = np.take(med, idx[1])
    keep = D.std(axis=0) > 1e-8
    D = D[:, keep]
    # Median-imputed columns are NaN-free by construction; assert so the failure is
    # a clear message here rather than an opaque XGBoost "inf in data" deep in libxgboost.
    if not np.isfinite(D).all():
        bad = np.where(~np.isfinite(D).all(axis=0))[0]
        raise AssertionError(f"non-finite descriptor columns survived imputation: {bad[:10]}")

    # Some RDKit descriptors are scale-invariant but astronomically large (Ipc ~ 1e14).
    # XGBoost rejects values beyond its internal gradient-index limit. They are useless
    # to a tree model anyway (ordered splits would never land on the informative region),
    # so drop columns by ABSOLUTE scale, not by variance.
    mag = np.abs(D).max(axis=0)
    n_dropped = int((mag > 1e6).sum())
    if n_dropped:
        print(f"  [featurize] dropped {n_dropped} descriptor columns with |value| > 1e6 "
              f"(max kept = {mag[mag <= 1e6].max():.3g})")
    D = D[:, mag <= 1e6]
    return np.hstack([np.asarray(fps, dtype=np.float32), D.astype(np.float32)])


def folds_for(ds, k=5, seed=1337):
    p = ROOT / f"5_outputs/split_cache/{ds}_scaffold_k{k}_seed{seed}.json"
    d = json.loads(p.read_text())
    return [(np.asarray(f["train"], int), np.asarray(f["test"], int)) for f in d["folds"]]


def run(X, y, folds, train_mask=None, tag=""):
    """OOF predictions; returns pooled R2 over the FULL set and fold-mean R2."""
    pred = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in folds:
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        m = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
            n_jobs=32, tree_method="hist", random_state=0,
        )
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        per_fold.append(r2_score(y[te], pred[te]))
    pooled = r2_score(y, pred)
    print(f"  {tag:34s} pooled_R2={pooled:.4f}  fold_mean={np.mean(per_fold):.4f}"
          f"+-{np.std(per_fold):.4f}", flush=True)
    return pooled, float(np.mean(per_fold)), pred


def main():
    out = {}
    for ds in ["Sheet2", "Sheet3"]:
        smis, y = load(ds)
        X = featurize(smis)
        folds = folds_for(ds)
        print(f"\n=== {ds}  n={len(y)}  features={X.shape[1]}  (ECFP4 2048 + RDKit)")
        pooled, fm, pred = run(X, y, folds, tag="ALL rows (neural protocol)")
        out[f"{ds}|all"] = dict(pooled=pooled, fold_mean=fm)

        if ds == "Sheet2":
            unc = y > 0
            print(f"  censoring: {(~unc).sum()}/{len(y)} rows have y<=0 "
                  f"({100*(~unc).mean():.1f}%)")
            pooled2, fm2, _ = run(X, y, folds, train_mask=unc,
                                  tag="train on UNCENSORED rows only")
            out[f"{ds}|uncensored_train"] = dict(pooled=pooled2, fold_mean=fm2)
            # and how well does a pure-chemistry model do on just the uncensored rows?
            sub = np.where(unc)[0]
            print(f"  (reference) uncensored-only subset: n={len(sub)} "
                  f"std={y[sub].std():.3f} vs full std={y.std():.3f}")

    (ROOT / "5_outputs/strong_baseline_probe.json").write_text(json.dumps(out, indent=2))
    print("\nwrote 5_outputs/strong_baseline_probe.json")


if __name__ == "__main__":
    main()
