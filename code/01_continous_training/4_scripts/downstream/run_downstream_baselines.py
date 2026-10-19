#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver: run the classical baselines over the frozen downstream split grid.

This is the counterpart to the MolTrain queue scripts. It reads the SAME frozen
split files the neural arms read (``downstream_splits/``) and writes results in
the SAME format (``downstream_results/*.json`` + ``downstream_predictions/*.npz``)
via ``run_matrix.run_sklearn_model``, so ``run_matrix.compare_runs`` can pair a
baseline against a neural run fold-by-fold without any adapter layer.

Usage:
    python 4_scripts/downstream/run_downstream_baselines.py \
        --dataset pKa --split butina --models ecfp4_rf ecfp4_xgb mlp_morgan

    # every seed of the frozen grid, one model
    python 4_scripts/downstream/run_downstream_baselines.py \
        --dataset pKa --split scaffold --models mlp_morgan --seeds 42 123 456 789 1011

    # evaluation 1 (TE) -- seeds default to that dataset's own frozen set
    python 4_scripts/downstream/run_downstream_baselines.py \
        --dataset Sheet2 --split scaffold --models ecfp4_rf ecfp4_xgb mlp_morgan
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines  # noqa: E402
import ds_registry  # noqa: E402
import run_matrix  # noqa: E402
import split_files  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(ds_registry.LOADERS))
    ap.add_argument("--split", required=True, choices=["scaffold", "butina", "random"])
    ap.add_argument("--models", nargs="+", required=True,
                    choices=sorted(baselines.REGISTRY))
    # Default is per (dataset, split) rather than one global list: Sheet2/Sheet3
    # were frozen at seeds {42, 1337, 2024} and asking for the plan's 5-seed grid
    # by default would just crash on the missing files. Pass --seeds to override.
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="partition seeds (default: the frozen set for this dataset/split)")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--target", default=None,
                    help="for multi-target datasets, evaluate only this column "
                         "(default: every target, which is 16 runs on MTR)")
    args = ap.parse_args()

    df, target_cols = ds_registry.load(args.dataset, verbose=True)
    if args.target:
        target_cols = [args.target]
    smiles = df["SMILES"].astype(str).tolist()
    seeds = args.seeds or split_files.seeds_for(args.dataset, args.split)

    # Fingerprints are a property of the dataset, computed once. Recomputing per
    # seed would be ~30 s of RDKit x 5 seeds for no reason.
    X_morgan = None
    X_atompair = None

    for model in args.models:
        factory = baselines.REGISTRY[model]
        if model == "xgb_atompair":
            if X_atompair is None:
                X_atompair = baselines.atompair_fingerprints(smiles)
                print(f"  atompair fingerprints: {X_atompair.shape}  seeds={seeds}")
            X = X_atompair
        else:
            if X_morgan is None:
                X_morgan = baselines.morgan_fingerprints(smiles)
                print(f"  morgan fingerprints: {X_morgan.shape}  seeds={seeds}")
            X = X_morgan
        for seed in seeds:
            folds = split_files.load_folds(args.dataset, args.split, seed,
                                           args.kfold, smiles=smiles)
            if model == "mlp_morgan":
                # The MLP is single-output by construction; multi-target would
                # need a shared trunk, which is a different model. Refuse rather
                # than silently looping and overwriting the same run_id.
                if len(target_cols) != 1:
                    raise SystemExit(
                        f"mlp_morgan is single-target; pass --target. "
                        f"{args.dataset} has {len(target_cols)} targets.")
            # The WHOLE target matrix, not column 0. `baselines`' multi-output arms
            # (RF/XGB) loop `for j in range(Y.shape[1])` and slice `_y_slice` per
            # output, so binding one column would train a 16-output-shaped model
            # against a 1-column target: RF/XGB would raise on the shape, and any
            # arm that survived would have every one of its 16 columns fitted to
            # assay 0 while `run_matrix` scored them against assays 0..15. That is
            # a wrong-but-plausible table, which is the failure mode to avoid.
            baselines.bind_target(df[target_cols].to_numpy(float))
            t0 = time.time()
            summary = run_matrix.run_sklearn_model(
                factory(X), df, target_cols, folds,
                dataset=args.dataset, model=model, split=args.split, seed=seed)
            if len(target_cols) == 1:
                r2 = summary["per_target"][target_cols[0]]["r2"]
                print(f"  [{model}/{args.split}/seed{seed}] fold-mean R2={r2:.4f}  "
                      f"({time.time() - t0:.1f}s)  -> {summary['run_id']}")
            else:
                # One line per assay: a fold-mean over 16 columns would describe no
                # assay at all (the safety columns are on a different scale from the
                # transfection ones), so there is no honest single headline here.
                print(f"  [{model}/{args.split}/seed{seed}] "
                      f"{len(target_cols)} targets  ({time.time() - t0:.1f}s)  "
                      f"-> {summary['run_id']}")
                for t in target_cols:
                    b = summary["per_target"][t]
                    flag = "  [near-constant: exclude from headline]" \
                        if t in ds_registry.MTR_LOW_VARIANCE else ""
                    print(f"      {t:34s} R2={b['r2']:+.4f}+-{b['r2_std']:.4f}{flag}")


if __name__ == "__main__":
    main()
