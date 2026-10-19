#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired comparison of the 3D-conformer fine-tune against the built-in-generator run.

Both arms are trained by finetune_moltrain.py on identical cached scaffold folds,
identical checkpoint, identical lr/epochs/early-stopping/seed. The only difference is
where the atom coordinates came from:

  baseline : MolTrain's inner_smi2coords(mode='fast') -> FLAT 2D for ~82% of rows
  3dconf   : pre-generated ETKDGv3 + MMFF94s conformers (100% verified 3D)

Reads each arm's cv.data (OOF predictions, z-scored) plus its target_scaler, inverts
the scaling, and reports per-fold R2 for both arms, the paired difference per fold,
and a paired t-test. Pairing is per FOLD, which is the only valid unit here: the two
arms share fold assignments, so the fold is the matched sample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score

REPO = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")

RUNS = {
    "baseline (MolTrain builtin)":
        REPO / "5_outputs/finetune_moltrain/"
               "Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed1337_lr5e-05_ep40",
    "3D conformers (ETKDGv3+MMFF94s)":
        REPO / "5_outputs/finetune_moltrain/"
               "Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed1337_lr5e-05_ep40_3dconf",
}


def oof_pred(run_dir: Path) -> np.ndarray:
    """Out-of-fold predictions on the MODELLING scale (log2 RLU).

    cv.data is stored z-scored: MolTrain writes `self.cv['pred']` BEFORE the
    inverse_transform that produces clf.cv_pred, so the scaler must be applied here.
    Reversing that exactly is why this reads target_scaler.ss rather than assuming
    either convention.
    """
    pred_z = np.asarray(joblib.load(run_dir / "cv.data")).ravel()
    scaler = joblib.load(run_dir / "target_scaler.ss")
    return scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()


def main() -> int:
    d = pd.read_excel(XLSX, sheet_name="Sheet2").dropna(subset=["SMILES", "RLU (log2)"])
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    folds = json.loads(
        (REPO / "5_outputs/split_cache/Sheet2_scaffold_k5_seed1337.json").read_text()
    )["folds"]

    preds, per_fold = {}, {}
    for name, run_dir in RUNS.items():
        if not (run_dir / "cv.data").exists():
            print(f"!! missing cv.data for {name} -> {run_dir}")
            print(f"   (run has not finished yet)")
            return 2
        p = oof_pred(run_dir)
        if len(p) != len(y):
            print(f"!! {name}: {len(p)} predictions vs {len(y)} labels")
            return 2
        preds[name] = p
        per_fold[name] = np.array(
            [r2_score(y[f["test"]], p[f["test"]]) for f in folds])

    print(f"Sheet2 scaffold seed1337, n={len(y)}, {len(folds)} folds")
    print(f"baseline predictions vs labels: mean={preds[list(RUNS)[0]].mean():.3f} "
          f"(labels mean={y.mean():.3f})")
    print()
    hdr = f"{'fold':>5} " + " ".join(f"{n[:22]:>22}" for n in RUNS) + f"{'delta':>10}"
    print(hdr)
    print("-" * len(hdr))
    names = list(RUNS)
    for k in range(len(folds)):
        row = " ".join(f"{per_fold[n][k]:>22.4f}" for n in names)
        delta = per_fold[names[1]][k] - per_fold[names[0]][k]
        print(f"{k:>5} {row} {delta:>10.4f}")
    print("-" * len(hdr))
    means = {n: float(per_fold[n].mean()) for n in names}
    stds = {n: float(per_fold[n].std(ddof=1)) for n in names}
    print(f"{'mean':>5} " + " ".join(f"{means[n]:>22.4f}" for n in names))
    print(f"{'std':>5} " + " ".join(f"{stds[n]:>22.4f}" for n in names))
    print()
    for n in names:
        print(f"  {n:36s} fold-mean R2 = {means[n]:.4f} +- {stds[n]:.4f}   "
              f"pooled = {r2_score(y, preds[n]):.4f}")

    diff = per_fold[names[1]] - per_fold[names[0]]
    t, p = ttest_rel(per_fold[names[1]], per_fold[names[0]])
    print(f"\npaired t-test over {len(folds)} folds (3D - baseline):")
    print(f"  mean delta = {diff.mean():+.4f}   t = {t:.3f}   p = {p:.4f}")
    print(f"  per-fold deltas: {[round(float(v), 4) for v in diff]}")
    if p < 0.05:
        print(f"  -> SIGNIFICANT at p<0.05")
    else:
        print(f"  -> not significant; note that folds from ONE partition are not "
              f"independent samples, so this p is anti-conservative either way")
    return 0


if __name__ == "__main__":
    sys.exit(main())
