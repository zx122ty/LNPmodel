#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cheap-feature information-ceiling probe for the LNP downstream task.

Question: how much of the fine-tuned LipidFM signal (Sheet2 scaffold s1337 pooled
R2 ~ 0.7) can be recovered from *2D cheap features* (Morgan FP + RDKit descriptors)?
If a small nonlinear model on cheap features matches the 3D pretrained backbone,
then the backbone's 3D geometry is not where the downstream signal lives, and
further pretraining investment on the 3D side has a low ceiling.

Evaluation must match the fine-tune protocol: identical cached scaffold folds
(5_outputs/split_cache), pooled R2 + fold-mean R2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")

# (sheet_name, target_column, log10); Sheet3's target gained an IGROV1_ prefix
# in the 2026-09-13 data fix, so both spellings are accepted.
SHEET3_TARGET = ("IGROV1_transfection_efficiency", "transfection_efficiency")
DATASETS = {
    "Sheet2": ("Sheet2", "RLU (log2)", False),
    "Sheet3": ("Sheet3", SHEET3_TARGET, True),
}


def load(dataset: str):
    sh, col, log10 = DATASETS[dataset]
    d = pd.read_excel(XLSX, sheet_name=sh)
    if isinstance(col, tuple):
        col = next((c for c in col if c in d.columns), None)
        if col is None:
            raise SystemExit(f"Sheet3 target column not found in {list(d.columns)}")
    d = d.dropna(subset=["SMILES", col]).reset_index(drop=True)
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d[col]).values.astype(float)
    if log10:
        y = np.log10(y)
    return smis, y


def rdkit_desc(smis):
    keys = None
    rows = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        v = Descriptors.CalcMolDescriptors(m)
        if keys is None:
            keys = list(v.keys())
        rows.append([v[k] if v[k] is not None else np.nan for k in keys])
    X = np.asarray(rows, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])
    keep = X.std(axis=0) > 1e-8
    return X[:, keep], [k for k, ok in zip(keys, keep) if ok]


def morgan(smis, radius=2, nbits=2048):
    rows = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        f = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        rows.append(np.asarray(f, dtype=np.float32))
    return np.asarray(rows)


def folds_for(dataset, split, k, seed):
    p = ROOT / f"5_outputs/split_cache/{dataset}_{split}_k{k}_seed{seed}.json"
    d = json.loads(p.read_text())
    key = "folds" if "folds" in d else None
    if key is None:
        # discover the fold container
        for kk, vv in d.items():
            if isinstance(vv, list) and vv and isinstance(vv[0], (list, dict)):
                key = kk
                break
    raw = d[key]
    out = []
    for f in raw:
        if isinstance(f, dict):
            out.append((np.asarray(f["train"]), np.asarray(f["test"])))
        else:
            out.append((np.asarray(f[0]), np.asarray(f[1])))
    return out


def evaluate(X, y, folds, model_fn):
    pred = np.zeros(len(y))
    per_fold = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = model_fn()
        m.fit(Xtr, y[tr])
        pred[te] = m.predict(Xte)
        per_fold.append(r2_score(y[te], pred[te]))
    return r2_score(y, pred), float(np.mean(per_fold)), float(np.std(per_fold))


def main():
    results = {}
    for dataset in ["Sheet2", "Sheet3"]:
        smis, y = load(dataset)
        folds = folds_for(dataset, "scaffold", 5, 1337)
        Xr, names = rdkit_desc(smis)
        Xm = morgan(smis)
        Xd = np.hstack([Xr, Xm])
        print(f"\n=== {dataset}  n={len(y)}  folds={len(folds)}  "
              f"rdkit={Xr.shape[1]} morgan={Xm.shape[1]}", flush=True)

        feats = {
            "rdkit200": Xr,
            "morgan2048": Xm,
            "rdkit+morgan": Xd,
        }
        models = {
            "RF300": lambda: RandomForestRegressor(300, n_jobs=32, random_state=0),
            "SVR-rbf": lambda: SVR(C=10.0, gamma="scale"),
        }
        for fname, X in feats.items():
            for mname, mfn in models.items():
                pooled, fm, fs = evaluate(X, y, folds, mfn)
                tag = f"{dataset}|{fname}|{mname}"
                results[tag] = dict(pooled=pooled, fold_mean=fm, fold_std=fs)
                print(f"  {fname:13s} {mname:8s} pooled_R2={pooled:.4f}  "
                      f"fold_mean={fm:.4f}+-{fs:.4f}", flush=True)

    out = ROOT / "5_outputs/cheap_feature_probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
