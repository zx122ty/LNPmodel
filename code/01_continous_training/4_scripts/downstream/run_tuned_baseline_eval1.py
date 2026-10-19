#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tuned traditional baseline on the evaluation-1 frozen grid, as a first-class arm.

Why this exists
---------------
``downstream/baselines.py`` fixes ``ecfp4_xgb`` at 600 trees / depth 6 and
``ecfp4_rf`` at 300 trees. Those are LIBRARY-DEFAULT-SCALE settings, chosen so
the arm answers "does the pretrained 3D encoder beat an MLP/RF/XGB on the same
input?" -- not "does a well-tuned tree model beat it?". The project's headline
claim has always been against the TUNED arm (CLAUDE.md: 0.6448), but
``traditional_baseline_tuned.py`` is Sheet2/scaffold-only and 15 folds, so on the
evaluation-1 25-fold plan grid the comparison was running against an untuned
opponent. This script closes that gap.

It does NOT replace ``run_downstream_baselines.py``. It adds a fourth arm
(``tuned_gbm``) that lives alongside it, so the untuned three stay exactly as
they were and the pair (untuned, tuned) becomes readable side by side.

Design decisions that are not free
----------------------------------
* **Same split files as every other arm.** ``split_files.load_folds`` on
  ``downstream_splits/``, not ``split_cache/``. The two are byte-identical for
  scaffold (verified 2026-09-19 for all 6 seed files) but only the former is the
  one the neural arms read, and "which file did you read" is precisely the class
  of bug this project keeps finding.

* **Selection is per (dataset, split), on seed 42 only, then frozen.** That is
  the protocol ``traditional_baseline_tuned.py`` already uses, kept so the two
  are comparable. It is also the protocol's one weakness: seed 42's number is
  mildly optimistic because selection saw those folds. The result file records
  ``selection_seed`` so a reader can drop that partition and re-rank.

* **The grid is the original 42 configs: 6 feature sets x (4 XGB + 3 LGB).** Kept
  identical to ``traditional_baseline_tuned.py`` on purpose, so the Sheet2/scaffold
  selection can be checked against the historical run rather than merely resembling
  it -- and it does: both pick ``desc+morgan_r3_2048|xgb`` with seed-42 R2 = 0.6509.
  A reduced grid would have made that check impossible for no real saving.

* **LightGBM is available and is reported.** It tied XGB within noise on Sheet2
  and is 3-5x faster, so if a reader wants the strongest classical arm it is not
  obviously XGB.

* **CPU time, not wall clock.** ``cost_probe.py`` established (坑 #14d) that wall
  clock here is a function of how busy the machine was -- 80x on the same command
  with bit-identical R2. Every fit reports ``time.process_time()``.

Usage
-----
    DRY_RUN=1 python 4_scripts/downstream/run_tuned_baseline_eval1.py
    python 4_scripts/downstream/run_tuned_baseline_eval1.py

    # one combo, for a spot check
    python 4_scripts/downstream/run_tuned_baseline_eval1.py --only Sheet2/scaffold

Idempotent: a (dataset, split, seed) cell that already has a result file is
skipped unless FORCE=1. Mirrors the guard the other two queues already have
(``06-*`` §5.4), which this script would otherwise have been the third one to
lack.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
import sys
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines  # noqa: E402
import ds_registry  # noqa: E402
import run_matrix  # noqa: E402
import split_files  # noqa: E402

ARM = "tuned_gbm"
SEEDS = (42, 123, 456, 789, 1011)
SELECT_SEED = 42
COMBOS = [("Sheet2", "scaffold"), ("Sheet2", "butina"),
          ("Sheet3", "scaffold"), ("Sheet3", "butina")]

N_WORKERS = int(os.environ.get("TB_NJOBS", "10"))
INNER_THREADS = int(os.environ.get("TB_THREADS", "2"))

# Same grid as traditional_baseline_tuned.py, minus nothing: the feature sets are
# what moved the number there, so all six are kept.
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

FEAT_CACHE = ROOT / "5_outputs/feat_cache"
FEAT_DIR = ROOT / "5_outputs/feat_cache_tuned"


def rdkit_descriptors(smis: list[str]) -> np.ndarray:
    """Same routine as traditional_baseline_tuned.rdkit_descriptors, cached per dataset.

    Imported by copy rather than by import: that module runs its whole experiment
    at import time (``main()`` is guarded, but ``DS``/``XLSX`` are module-level
    constants read by the helpers), so importing it would drag Sheet2's paths
    into a Sheet3 run.
    """
    from rdkit.Chem import Descriptors
    from rdkit import Chem
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


def morgan(smis: list[str], radius: int, nbits: int) -> np.ndarray:
    from rdkit.Chem import AllChem
    from rdkit import Chem
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.asarray(
        [np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), dtype=np.float32)
         for s in smis], dtype=np.float32)


def _cache(ds: str, name: str, fn) -> np.ndarray:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    p = FEAT_DIR / f"{ds}_{name}.npy"
    if p.exists():
        return np.load(p)
    X = fn()
    np.save(p, X)
    return X


def feature_sets(ds: str, smis: list[str]) -> dict:
    F = {
        "morgan_r2_2048": _cache(ds, "morgan_r2_2048", lambda: morgan(smis, 2, 2048)),
        "morgan_r3_2048": _cache(ds, "morgan_r3_2048", lambda: morgan(smis, 3, 2048)),
        "morgan_r2_1024": _cache(ds, "morgan_r2_1024", lambda: morgan(smis, 2, 1024)),
        "desc": _cache(ds, "rdkit_desc", lambda: rdkit_descriptors(smis)),
    }
    return {
        "morgan_r2_2048": F["morgan_r2_2048"],
        "morgan_r3_2048": F["morgan_r3_2048"],
        "morgan_r2_1024": F["morgan_r2_1024"],
        "desc": F["desc"],
        "desc+morgan_r2_2048": np.hstack([F["desc"], F["morgan_r2_2048"]]),
        "desc+morgan_r3_2048": np.hstack([F["desc"], F["morgan_r3_2048"]]),
    }


def _fit_one(kind: str, params: dict, X: np.ndarray, y: np.ndarray,
             folds: list) -> tuple[float, list[float], np.ndarray]:
    from sklearn.metrics import r2_score
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


def run_combo(ds: str, split: str, force: bool) -> None:
    from joblib import Parallel, delayed

    df, target_cols = ds_registry.load(ds, verbose=False)
    y = df[target_cols[0]].to_numpy(float)
    smis = df["SMILES"].astype(str).str.strip().tolist()
    X_all = feature_sets(ds, smis)

    todo = [s for s in SEEDS
            if force or not (ROOT / f"5_outputs/downstream_results/{ds}|{ARM}|{split}|seed{s}|k5.json").exists()]
    if not todo:
        print(f"[{ds}/{split}] all {len(SEEDS)} seeds present -- SKIP (FORCE=1 to rerun)")
        return
    print(f"[{ds}/{split}] seeds to run: {todo}  (present: "
          f"{[s for s in SEEDS if s not in todo]})")

    # ---- stage 1: hyperparameter selection, on SELECT_SEED only ----
    sel_folds = split_files.load_folds(ds, split, SELECT_SEED, 5, smiles=smis)
    jobs = [(fs, kind, pi, p)
            for fs in X_all
            for kind, grid in (("xgb", XGB_GRID), ("lgb", LGB_GRID))
            for pi, p in enumerate(grid)]
    t0, c0 = time.time(), time.process_time()
    out = Parallel(n_jobs=N_WORKERS, verbose=0)(
        delayed(_fit_one)(kind, p, X_all[fs], y, sel_folds)
        for fs, kind, pi, p in jobs)
    # Wall clock, not process_time: the fits run in joblib WORKER PROCESSES, whose
    # CPU time the parent's clock never sees. Reporting process_time() here gives
    # a confident "cpu=0s" -- the same class of unportable/absent measurement that
    # 坑 #14d is about, so it is labelled for what it is instead.
    sel_wall = time.time() - t0
    scored: dict = {}
    for (fs, kind, pi, p), (mean_r2, _pf, _pred) in zip(jobs, out):
        scored.setdefault(f"{fs}|{kind}", []).append((mean_r2, pi, p))
    best = {}
    for key, rows in sorted(scored.items()):
        rows.sort(reverse=True, key=lambda r: r[0])
        best[key] = dict(params=rows[0][2], sel_r2=round(rows[0][0], 4),
                         grid_index=rows[0][1])
    sel_key = max(best, key=lambda k: best[k]["sel_r2"])
    print(f"[{ds}/{split}] selection: {len(jobs)} configs on seed{SELECT_SEED}, "
          f"wall={sel_wall:.0f}s  winner={sel_key} selR2={best[sel_key]['sel_r2']:.4f}")

    # ---- stage 2: the winning config, refit on every plan seed ----
    fs, kind = sel_key.split("|")
    X = X_all[fs]
    baselines.bind_target(y)
    for seed in todo:
        rid = run_matrix.run_id(ds, ARM, split, seed, 5)
        path = ROOT / f"5_outputs/downstream_results/{rid}.json"
        if path.exists() and not force:
            print(f"  [{rid}] exists -- SKIP")
            continue
        folds = split_files.load_folds(ds, split, seed, 5, smiles=smis)
        tc, c1 = time.time(), time.process_time()
        # The refit must reproduce the SELECTION fold list exactly on SELECT_SEED,
        # or the reported selection score and the reported test score describe
        # different partitions. Cheap to assert, catastrophic to skip.
        if seed == SELECT_SEED:
            same = all(np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
                       for a, b in zip(folds, sel_folds))
            if not same:
                raise SystemExit(f"{rid}: fold list differs from the selection pass")
        # Fit every fold once, then hand ``run_sklearn_model`` a lookup into the
        # resulting OOF vector. Doing it inside the closure would refit the whole
        # fold list once per fold (5x the work) and, worse, would make the stored
        # OOF depend on the closure's fold list rather than on the one this cell
        # was scored with.
        _, per_fold, oof = _fit_one(kind, best[sel_key]["params"], X, y, folds)
        summary = run_matrix.run_sklearn_model(
            lambda tr, te, _o=oof: _o[np.asarray(te, int)].reshape(-1, 1),
            df, target_cols, folds,
            dataset=ds, model=ARM, split=split, seed=seed)
        summary["_tuned"] = {
            "feature_set": fs, "booster": kind,
            "params": best[sel_key]["params"],
            "selection_seed": SELECT_SEED,
            "selection_r2": best[sel_key]["sel_r2"],
            "grid": {k: v for k, v in best.items()},
            "cpu_sec_this_cell": round(time.process_time() - c1, 1),
            "note": ("Hyperparameters selected on seed "
                     f"{SELECT_SEED} on THIS (dataset, split); that partition's "
                     "number is mildly optimistic. Drop seed "
                     f"{SELECT_SEED} to re-rank honestly."),
        }
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        r2 = summary["per_target"][target_cols[0]]["r2"]
        print(f"  [{rid}] R2={r2:.4f}  cpu={summary['_tuned']['cpu_sec_this_cell']:.0f}s "
              f"wall={time.time()-tc:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset like Sheet2/scaffold")
    args = ap.parse_args()
    force = os.environ.get("FORCE") == "1"
    dry = os.environ.get("DRY_RUN") == "1"
    combos = COMBOS
    if args.only:
        combos = [tuple(x.split("/")) for x in args.only]
    print(f"=== tuned_gbm arm on the eval-1 plan grid ===")
    print(f"    combos={combos}  seeds={list(SEEDS)}  select_seed={SELECT_SEED}")
    print(f"    workers={N_WORKERS} x {INNER_THREADS} threads  FORCE={force}")
    if dry:
        for ds, sp in combos:
            missing = [s for s in SEEDS
                       if not (ROOT / f"5_outputs/downstream_results/{ds}|{ARM}|{sp}|seed{s}|k5.json").exists()]
            print(f"    [DRY] {ds}/{sp}: would run {len(missing)} seeds {missing}")
        print("    (DRY_RUN=1 -- nothing launched)")
        return
    for ds, sp in combos:
        run_combo(ds, sp, force)
    print("\nALL DONE. Now re-run the paired tables with the fourth arm:")
    for ds, sp in COMBOS:
        print(f"  python 4_scripts/downstream/report_table.py --dataset {ds} --split {sp} \\")
        print(f"    --models lipidfm_v5 {ARM} ecfp4_rf ecfp4_xgb mlp_morgan \\")
        print(f"    --seeds {' '.join(str(s) for s in SEEDS)}")


if __name__ == "__main__":
    main()
