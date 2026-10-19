#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The constant-predictor control: what does "predict the train mean" score?

Why this exists as its own script
--------------------------------
`prompt/26-09-16/04-*` established that a scaffold split can be **unusable** even
when it passes every structural check: if one fold is filled by a single scaffold,
the model is scoring a chemical space it never saw, and a constant predictor does
just as well. The discriminating quantity there is *max group / mean fold size*
(threshold 1.0) -- but that is only a NECESSARY condition. The sufficient
condition is measuring the constant predictor itself, and on Sheet3 / MTR it is
dramatic: -0.83 and -3.93 versus -0.04 on pKa.

That measurement was previously made ad hoc, so the numbers in
`05-pKa评估结果.md` §2.2 could not be reproduced by a reviewer. This script makes
it a first-class, re-runnable check over the FROZEN folds.

What it computes
----------------
For every fold, predict the TRAIN fold's mean for every test molecule, then score
the test fold. That is the weakest possible model that still uses the labels; any
real model must beat it by a wide margin or the fold is not measuring learning.

Note the fold-mean R2 of this predictor is not exactly 0 even on a good split:
it is negative by the amount the train mean misses the test mean, which is
nonzero whenever the partition is not label-balanced. A large negative value is
the signature of a mean shift between train and test -- exactly the failure mode
that makes a scaffold split unusable on a scaffold-converged library.

Usage:
    python 4_scripts/downstream/constant_predictor.py                 # every dataset/split
    python 4_scripts/downstream/constant_predictor.py --dataset pKa --split butina
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds_registry  # noqa: E402
import split_files  # noqa: E402

OUT = ROOT / "5_outputs/downstream_results/constant_predictor.json"


def run(ds_key: str, split: str, seeds: list[int], kfold: int = 5) -> dict:
    df, target_cols = ds_registry.load(ds_key, verbose=False)
    smiles = df["SMILES"].astype(str).tolist()
    y = df[target_cols[0]].to_numpy(float)

    per_seed, fold_r2_all = [], []
    for seed in seeds:
        folds = split_files.load_folds(ds_key, split, seed, kfold, smiles=smiles)
        r2s = []
        for tr, te in folds:
            tr = np.asarray(tr, int)
            te = np.asarray(te, int)
            # The only prediction available to a model with no features: the
            # training fold's mean. Computed from train ONLY -- using the full
            # column would be a leak and would flatter the control.
            r2s.append(float(_r2(y[te], np.full(len(te), y[tr].mean()))))
        per_seed.append(float(np.mean(r2s)))
        fold_r2_all.extend(r2s)

    return {
        "dataset": ds_key, "split": split, "seeds": list(seeds), "kfold": kfold,
        "n": len(y),
        "fold_mean_r2": float(np.mean(fold_r2_all)),
        # Two spreads, because they answer different questions and the project
        # has quoted both. `fold_std_r2` is the spread across individual folds
        # (what a paired test sees); `per_seed_std` is the spread of the seed
        # means (what `05-pKa评估结果.md` §2.2 quotes, as "± 0.0023"). Reporting
        # only one of them next to the other's number is how a reader ends up
        # comparing 0.0023 with 0.034 and concluding something changed.
        "fold_std_r2": float(np.std(fold_r2_all)),
        "per_seed_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        "n_folds": len(fold_r2_all),
        "per_seed_means": [round(v, 4) for v in per_seed],
        "worst_fold_r2": float(np.min(fold_r2_all)),
        "target": target_cols[0],
    }


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--json-out", type=Path, default=OUT)
    args = ap.parse_args()

    results = {}
    for ds_key in ([args.dataset] if args.dataset else sorted(ds_registry.LOADERS)):
        for split in ([args.split] if args.split
                      else split_files.SPLITS_BY_DATASET[ds_key]):
            seeds = split_files.seeds_for(ds_key, split)
            r = run(ds_key, split, seeds)
            results[f"{ds_key}|{split}"] = r
            flag = ""
            # 04-* uses -0.10 as the "this split is measuring nothing" scale: on
            # Sheet3/MTR the constant predictor beats every real model at -0.83
            # and -3.93. Anything above that bar is a warning, not a verdict.
            if r["fold_mean_r2"] < -0.15:
                flag = "  ⚠️ constant predictor far below 0 -- train/test mean shift"
            print(f"{ds_key:<7} {split:<10} n={r['n']:<5} {r['n_folds']:>2} folds  "
                  f"fold-mean R2 = {r['fold_mean_r2']:>+8.4f} +- {r['fold_std_r2']:.4f}  "
                  f"(worst fold {r['worst_fold_r2']:+.3f}){flag}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
