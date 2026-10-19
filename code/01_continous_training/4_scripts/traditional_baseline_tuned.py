#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tuned traditional baseline vs the neural fine-tune, on paired repeated CV.

Purpose: settle whether "fingerprint + gradient-boosted trees" actually beats the 3D
neural model here. The earlier probe (strong_baseline_probe.py) scored 0.649 with
DEFAULT XGBoost parameters -- below the neural 0.6975 -- so the stronger claim was
never tested at the traditional model's best. This gives it a fair hearing:

  * fingerprint radius 2 and 3, 1024 and 2048 bits
  * XGBoost AND LightGBM
  * a real hyperparameter search, not library defaults
  * repeated 5-fold: 3 scaffold partitions (42 / 1337 / 2024) = 15 folds

Protocol matches finetune_moltrain.py exactly (its cached scaffold folds, fold-mean
R2 on log2 RLU), so every number is pairable with the neural runs on disk.

Parallelism: one process per (feature_set, model, config). Each fit gets n_jobs=2 so
the pool does not fight the GPU fine-tune for cores. A single config is ~90 s of CPU
work, so serialising 49 of them costs over an hour; spreading them across workers
brings the search down to minutes.

Honesty note: hyperparameters are selected on the seed-42 partition and then applied
unchanged to seeds 1337 and 2024. Seed 42's number is therefore mildly optimistic
(selection saw those folds); 1337 and 2024 are clean.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import r2_score

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
OUT = ROOT / "5_outputs/traditional_baseline_tuned.json"
FEAT_CACHE = ROOT / "5_outputs/feat_cache"

DS = "Sheet2"
TARGET_COL = "RLU (log2)"
REPEATS = (42, 1337, 2024)
SELECT_SEED = 42

N_WORKERS = int(os.environ.get("TB_NJOBS", "10"))
INNER_THREADS = 2

XGB_GRID = [
    dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
         colsample_bytree=0.5, reg_lambda=1.0),
    dict(n_estimators=1200, max_depth=8, learning_rate=0.03, subsample=0.7,
         colsample_bytree=0.3, reg_lambda=2.0),
    dict(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
         colsample_bytree=0.7, reg_lambda=0.5),
    dict(n_estimators=900, max_depth=5, learning_rate=0.05, subsample=0.8,
         colsample_bytree=0.2, reg_lambda=1.0, min_child_weight=3),
]
LGB_GRID = [
    dict(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8,
         colsample_bytree=0.5, reg_lambda=1.0),
    dict(n_estimators=1200, num_leaves=31, learning_rate=0.03, subsample=0.7,
         colsample_bytree=0.3, reg_lambda=2.0),
    dict(n_estimators=300, num_leaves=15, learning_rate=0.1, subsample=0.9,
         colsample_bytree=0.7, reg_lambda=0.5),
]


# --------------------------------------------------------------------------- features
def rdkit_descriptors(smis):
    """Ipc reaches ~1e14 and trips XGBoost's internal gradient-index limit, so
    oversized columns are dropped by ABSOLUTE SCALE (scale-invariant, useless to
    ordered splits anyway)."""
    keys, rows = None, []
    for s in smis:
        v = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))
        if keys is None:
            keys = list(v.keys())
        rows.append([v[k] if v[k] is not None else np.nan for k in keys])
    D = np.asarray(rows, dtype=np.float64)
    D[~np.isfinite(D)] = np.nan
    med = np.nanmedian(D, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(~np.isfinite(D))
    D[idx] = np.take(med, idx[1])
    D = D[:, D.std(axis=0) > 1e-8]
    D = D[:, np.abs(D).max(axis=0) <= 1e6]
    return D.astype(np.float32)


def morgan(smis, radius, nbits):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.asarray(
        [np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), dtype=np.float32)
         for s in smis], dtype=np.float32)


def _cache(name, fn):
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    p = FEAT_CACHE / f"{DS}_{name}.npy"
    if p.exists():
        return np.load(p)
    X = fn()
    np.save(p, X)
    return X


def folds_for(seed, k=5):
    p = ROOT / f"5_outputs/split_cache/{DS}_scaffold_k{k}_seed{seed}.json"
    d = json.loads(p.read_text())
    return [(np.asarray(f["train"], int), np.asarray(f["test"], int)) for f in d["folds"]]


def _fit_one(kind, params, X, y, folds):
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(tree_method="hist", n_jobs=INNER_THREADS,
                             random_state=0, **params)
    else:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_jobs=INNER_THREADS, random_state=0, verbose=-1, **params)
    pred = np.zeros(len(y))
    per_fold = []
    for tr, te in folds:
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        per_fold.append(float(r2_score(y[te], pred[te])))
    return float(np.mean(per_fold)), per_fold, pred


def main():
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", TARGET_COL])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d[TARGET_COL]).values.astype(float)
    print(f"{DS}: n={len(y)}  target=log2 RLU  fold-mean R2 protocol", flush=True)

    F = {
        "morgan_r2_2048": _cache("morgan_r2_2048", lambda: morgan(smis, 2, 2048)),
        "morgan_r3_2048": _cache("morgan_r3_2048", lambda: morgan(smis, 3, 2048)),
        "morgan_r2_1024": _cache("morgan_r2_1024", lambda: morgan(smis, 2, 1024)),
        "desc": _cache("rdkit_desc", lambda: rdkit_descriptors(smis)),
    }
    FEATURE_SETS = {
        "morgan_r2_2048": F["morgan_r2_2048"],
        "morgan_r3_2048": F["morgan_r3_2048"],
        "morgan_r2_1024": F["morgan_r2_1024"],
        "desc": F["desc"],
        "desc+morgan_r2_2048": np.hstack([F["desc"], F["morgan_r2_2048"]]),
        "desc+morgan_r3_2048": np.hstack([F["desc"], F["morgan_r3_2048"]]),
    }
    for k, v in FEATURE_SETS.items():
        print(f"  {k:22s} {v.shape}", flush=True)

    sel_folds = folds_for(SELECT_SEED)
    jobs = [(fs, kind, pi, params)
            for fs in FEATURE_SETS
            for kind, grid in (("xgb", XGB_GRID), ("lgb", LGB_GRID))
            for pi, params in enumerate(grid)]

    print(f"\n=== selection on seed {SELECT_SEED}: {len(jobs)} configs, "
          f"{N_WORKERS} workers x {INNER_THREADS} threads ===", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(_fit_one)(kind, params, FEATURE_SETS[fs], y, sel_folds)
        for fs, kind, pi, params in jobs
    )
    scored = {}
    for (fs, kind, pi, params), (mean_r2, _pf, _pred) in zip(jobs, out):
        key = f"{fs}|{kind}"
        scored.setdefault(key, []).append((mean_r2, pi, params))
    print(f"selection done in {time.time()-t0:.0f}s", flush=True)

    best = {}
    for key, rows in sorted(scored.items()):
        rows.sort(reverse=True, key=lambda r: r[0])
        best[key] = dict(params=rows[0][2], sel_r2=rows[0][0], grid_index=rows[0][1])
        print(f"  {key:32s} best grid#{rows[0][1]}  selR2={rows[0][0]:.4f}  "
              f"(all: {[round(r[0],4) for r in rows]})", flush=True)

    # ---- stage 2: apply each selected config to every partition ----
    print(f"\n=== evaluation on partitions {REPEATS} ===", flush=True)
    ev_jobs = []
    for key, info in best.items():
        fs, kind = key.split("|")
        for seed in REPEATS:
            ev_jobs.append((key, fs, kind, info, seed))
    t0 = time.time()
    ev_out = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(_fit_one)(kind, info["params"], FEATURE_SETS[fs], y, folds_for(seed))
        for _key, fs, kind, info, seed in ev_jobs
    )
    print(f"evaluation done in {time.time()-t0:.0f}s", flush=True)

    results = {}
    for (key, fs, kind, info, seed), (mean_r2, pf, _pred) in zip(ev_jobs, ev_out):
        r = results.setdefault(key, {
            "feature_set": fs, "model": kind, "params": info["params"],
            "grid_index": info["grid_index"], "per_seed": {},
        })
        r["per_seed"][str(seed)] = dict(mean=round(mean_r2, 4),
                                        folds=[round(v, 4) for v in pf])

    print("\n=== ranking (mean of 15 folds = 3 partitions x 5) ===")
    for key, r in results.items():
        allf = [v for s in REPEATS for v in r["per_seed"][str(s)]["folds"]]
        r["r2_mean_15"] = round(float(np.mean(allf)), 4)
        r["r2_std_15"] = round(float(np.std(allf, ddof=1)), 4)
        r["n_folds"] = len(allf)
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["r2_mean_15"])
    for key, r in ranked:
        print(f"  {r['r2_mean_15']:.4f} +- {r['r2_std_15']:.4f}  {key:32s} "
              f"per-seed {[r['per_seed'][str(s)]['mean'] for s in REPEATS]}")

    OUT.write_text(json.dumps({
        "dataset": DS, "target": TARGET_COL, "repeats": list(REPEATS),
        "selection_seed": SELECT_SEED,
        "caveat": ("Hyperparameters selected on seed 42; that partition's number is "
                   "mildly optimistic. Seeds 1337/2024 are clean."),
        "results": results,
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
