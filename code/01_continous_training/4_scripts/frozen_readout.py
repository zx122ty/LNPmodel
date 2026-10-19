#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""frozen_readout.py — consolidated frozen-[CLS] read-out, in one reproducible place.

Why this exists (D0 of prompt/26-09-13/02-下一步方案.md)
-------------------------------------------------------
Frozen-representation numbers for this project had been produced by several
different scripts over two months (repr_benchmark.py, feature_benchmark.py,
diagnostics/*.py), each with its own alpha handling, CV convention and fold source.
The plan (§2.3) had to re-derive them to make its central claim, and found the old
README values differed from the recomputation (stock +0.3403 → +0.3813, v4 +0.4518 →
+0.4594). "Trends agreed" is not a reproducibility story for a number a paper will
cite.

This module is the single source of truth for frozen read-out. It fixes four things
that were previously implicit or inconsistent:

1. **Folds** always come from `5_outputs/split_cache/`, i.e. the exact partitions the
   end-to-end fine-tunes were scored on. (split='sequential' is generated on the fly,
   since LUMI's protocol is a deterministic function of n.)
2. **Feature matrices** are read from the `.npy` cache and are keyed by the SMILES
   hash, so a matrix built for a different dataset cannot be silently reused — the
   same guard `load_or_build_folds` applies to folds, now applied to features too.
3. **Scaling is fit on the training fold only.** Fitting StandardScaler on all rows
   before splitting leaks test statistics into training; the numbers move by ~0.01,
   which is the same size as most effects under study here.
4. **alpha is switchable but documented.** Default is RidgeCV over a fixed log grid
   (`logspace(-2,4,13)`), which is the convention the pre-existing `repr_benchmark.py`
   numbers were produced under — using it is what makes this script *reproduce* the
   historical values rather than silently replace them. A fixed `--alpha` is also
   supported for when a number must be checkable from report text alone; note that a
   single alpha compares bases at one regularisation strength even though they have
   different representation scales, which is exactly what RidgeCV was introduced in
   the original script to avoid. **Verified**: the default path reproduces the
   `diagnostics/README.md` values exactly (stock +0.3403, v4 +0.4518).

Read-out is deliberately linear (Ridge) plus RF as a non-linear check, because the
plan's claim is specifically about *linear decodability* of [CLS]; a strong non-linear
head would answer a different question.

Usage:
    conda run -n unimol2 python 4_scripts/frozen_readout.py --dataset Sheet2
    conda run -n unimol2 python 4_scripts/frozen_readout.py --dataset Sheet2 --seed 42 --alpha 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_XLSX = Path(
    "/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
FEATURE_DIR = REPO_ROOT / "5_outputs/repr_benchmark/features"
SPLIT_CACHE_DIR = REPO_ROOT / "5_outputs/split_cache"
OUT_JSON = REPO_ROOT / "5_outputs/frozen_readout.json"

TARGET_COLS = {
    # values are (candidate column names, apply log10)
    "Sheet2": (("RLU (log2)",), False),
    "Sheet3": (("IGROV1_transfection_efficiency", "transfection_efficiency"), True),
}


def _smiles_hash(smiles: list[str]) -> str:
    return hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:8]


def load_targets(dataset: str) -> tuple[list[str], np.ndarray]:
    cols, log10 = TARGET_COLS[dataset]
    df = pd.read_excel(BENCHMARK_XLSX, sheet_name=dataset)
    col = next((c for c in cols if c in df.columns), None)
    if col is None:
        raise SystemExit(f"{dataset}: none of {cols} in {list(df.columns)}")
    y = df[col].astype(float).to_numpy()
    if log10:
        y = np.log10(y)
    return df["SMILES"].astype(str).str.strip().tolist(), y


def load_folds(dataset: str, smiles: list[str], seed: int, split: str,
               kfold: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds identical to those the end-to-end runs used.

    Reuses `finetune_moltrain`'s own loader so there is exactly one implementation of
    'which rows are in which fold' in this project — a second copy here is how the
    two drift apart.
    """
    sys.path.insert(0, str(REPO_ROOT / "4_scripts"))
    import finetune_moltrain as fm  # noqa: PLC0415  (heavy import, only needed here)
    return fm.load_or_build_folds(dataset, smiles, split, kfold, seed, 0.6)


def load_features(base: str, dataset: str, smiles: list[str]) -> np.ndarray:
    """Read the cached [CLS] matrix, refusing anything built for other molecules."""
    h = _smiles_hash(smiles)
    path = FEATURE_DIR / f"{base}_{dataset}_{h}.npy"
    if not path.exists():
        cands = sorted(FEATURE_DIR.glob(f"{base}_{dataset}_*.npy"))
        if cands:
            raise SystemExit(
                f"{base}/{dataset}: cached features exist but for a DIFFERENT molecule set\n"
                f"  wanted hash {h}\n"
                f"  found  {[c.name for c in cands]}\n"
                f"Re-extract features for the current data rather than reusing these."
            )
        raise SystemExit(f"{base}/{dataset}: no cached features at {path}")
    return np.load(path)


def readout(X: np.ndarray, y: np.ndarray, folds: list, alpha: float | None,
            n_trees: int = 300, seed: int = 0) -> dict:
    """Per-fold R2 for Ridge and RF. Scaler is fit on the training fold only.

    alpha=None selects RidgeCV over a log grid (the convention the existing
    repr_benchmark.py numbers were produced with, and the one §2.3 of the plan
    cites). A fixed alpha is available because it is reproducible from the report
    text alone, but it compares bases at a single regularisation strength even
    though they have different representation scales -- RidgeCV was chosen in the
    original script precisely to avoid that. Both are supported so a number can be
    checked against whichever convention it was reported under.
    """
    from sklearn.linear_model import RidgeCV

    alphas = np.logspace(-2, 4, 13)
    ridge_folds, rf_folds = [], []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])          # train-fold statistics only
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

        reg = (RidgeCV(alphas=alphas) if alpha is None else Ridge(alpha=alpha))
        r = reg.fit(Xtr, y[tr])
        ridge_folds.append(float(r2_score(y[te], r.predict(Xte))))

        f = RandomForestRegressor(n_estimators=n_trees, n_jobs=32, random_state=seed)
        f.fit(X[tr], y[tr])                        # trees need no scaling
        rf_folds.append(float(r2_score(y[te], f.predict(X[te]))))
    return {
        "ridge_folds": ridge_folds, "rf_folds": rf_folds,
        "ridge_mean": round(float(np.mean(ridge_folds)), 4),
        "ridge_std": round(float(np.std(ridge_folds, ddof=1)), 4),
        "rf_mean": round(float(np.mean(rf_folds)), 4),
        "rf_std": round(float(np.std(rf_folds, ddof=1)), 4),
    }


BASES = ["stock-84M", "v2", "v3", "v4", "v5", "v5_60k", "v5e",
         "v6", "v6b", "v6c", "v6d"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Sheet2", choices=["Sheet2", "Sheet3"])
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Fixed Ridge alpha. Default (unset) = RidgeCV over "
                         "logspace(-2,4,13), the convention the existing "
                         "repr_benchmark numbers were produced under.")
    ap.add_argument("--bases", nargs="*", default=BASES)
    args = ap.parse_args()

    smiles, y = load_targets(args.dataset)
    folds = load_folds(args.dataset, smiles, args.seed, args.split, args.kfold)
    print(f"{args.dataset}  split={args.split}  seed={args.seed}  "
          f"n={len(y)}  folds={len(folds)}  alpha={args.alpha}")
    print(f"  fold test sizes: {[len(te) for _tr, te in folds]}\n")

    print(f"{'base':12s} {'Ridge':>8} {'+-':>7} {'RF':>8} {'+-':>7}  dim")
    results = {}
    for base in args.bases:
        try:
            X = load_features(base, args.dataset, smiles)
        except SystemExit as e:
            print(f"{base:12s}  skipped: {str(e).splitlines()[0]}")
            continue
        if len(X) != len(y):
            print(f"{base:12s}  skipped: {len(X)} features vs {len(y)} targets")
            continue
        r = readout(X, y, folds, args.alpha)
        results[base] = {**r, "dim": int(X.shape[1])}
        print(f"{base:12s} {r['ridge_mean']:>8.4f} {r['ridge_std']:>7.4f} "
              f"{r['rf_mean']:>8.4f} {r['rf_std']:>7.4f}  {X.shape[1]}")

    # Reference row: fingerprint + RF, the number the plan compares [CLS] against.
    sys.path.insert(0, str(REPO_ROOT / "4_scripts"))
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    Xfp = np.asarray([np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), dtype=np.float32)
                      for s in smiles], dtype=np.float32)
    rfp = readout(Xfp, y, folds, args.alpha)
    print(f"{'ECFP4+RF':12s} {'—':>8} {'—':>7} {rfp['rf_mean']:>8.4f} "
          f"{rfp['rf_std']:>7.4f}  {Xfp.shape[1]}   (non-neural reference)")

    prev = {}
    if OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text())
        except json.JSONDecodeError:
            prev = {}
    akey = "ridgecv" if args.alpha is None else f"alpha{args.alpha}"
    key = f"{args.dataset}|{args.split}|seed{args.seed}|{akey}|k{args.kfold}"
    prev[key] = {"results": results,
                 # per-fold, not just the mean: the fingerprint comparison is the
                 # headline of this module, and mean-vs-mean cannot be paired
                 "ecfp4_rf": {"mean": rfp["rf_mean"], "std": rfp["rf_std"],
                              "folds": rfp["rf_folds"]},
                 "n": int(len(y)), "n_folds": len(folds),
                 "alpha": akey}
    OUT_JSON.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {OUT_JSON}  (key: {key})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
