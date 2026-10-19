#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do the two arms carry complementary signal? Test every ensemble on paired CV.

Two candidate pairings, both worth checking for the same reason -- if two models have
correlated errors below ~0.9, averaging them usually beats either one:

  1. NEURAL 2D + NEURAL 3D  : same backbone fine-tuned on built-in (mostly flat) vs
     ETKDGv3 conformers. Residual correlation measured at 0.80 on one partition.
  2. NEURAL + TRADITIONAL   : the 3D fine-tune against tuned ECFP4/RDKit + XGBoost.

Reporting rules that matter here:
  * Averaging happens on the PREDICTION scale (log2 RLU), not on R2 -- averaging
    per-fold R2 values is not an ensemble and would only dilute the spread.
  * Every arm must use the SAME fold assignment, so all comparisons are paired.
  * The headline is the paired per-fold difference vs the best single arm, with a
    paired t-test over 15 folds (3 partitions x 5).
  * Run-to-run variation on this benchmark is ~0.025 R2, so a +0.01 gains is only
    meaningful if it holds across partitions -- which is why this is repeated CV.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
SEEDS = (42, 1337, 2024)
DS = "Sheet2"

EXP = ("Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed{}"
       "_lr5e-05_ep40")


def load_oof(seed: int, arm3d: bool):
    """OOF predictions on the modelling scale (log2 RLU), or None if the run is absent."""
    p = ROOT / "5_outputs/finetune_moltrain" / (EXP.format(seed) + ("_3dconf" if arm3d else ""))
    if not (p / "cv.data").exists():
        return None
    pred_z = np.asarray(joblib.load(p / "cv.data")).ravel()
    scaler = joblib.load(p / "target_scaler.ss")
    return scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()


def labels_and_folds(seed: int):
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    folds = json.loads(
        (ROOT / f"5_outputs/split_cache/{DS}_scaffold_k5_seed{seed}.json").read_text()
    )["folds"]
    return y, folds


def main():
    print(f"{DS}, scaffold partitions {SEEDS}, fold-mean R2 on log2 RLU\n")
    per_fold = {}          # arm name -> list of fold R2
    corr = {}              # pair name -> list of per-seed residual correlations
    trad = np.load(ROOT / "5_outputs/traditional_oof.npz") if (
        ROOT / "5_outputs/traditional_oof.npz").exists() else None

    for seed in SEEDS:
        y, folds = labels_and_folds(seed)
        preds = {}
        for name, is3d in (("neural_2d", False), ("neural_3d", True)):
            p = load_oof(seed, is3d)
            if p is not None:
                preds[name] = p
        if trad is not None and f"pred_{seed}" in trad:
            preds["traditional"] = trad[f"pred_{seed}"]

        if len(preds) < 2:
            # still record the single arm so partial progress is visible
            for name, pred in preds.items():
                per_fold.setdefault(name, []).extend(
                    float(r2_score(y[f["test"]], pred[f["test"]])) for f in folds)
            continue

        # Residual correlation between every pair -- the diagnostic that decides
        # whether an ensemble can work at all.
        names = sorted(preds)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                r = pearsonr(y - preds[a], y - preds[b])[0]
                corr.setdefault(f"{a}~{b}", []).append(float(r))

        # Ensembles, built per seed on that partition's folds. Averaging the
        # PREDICTIONS is the operation that actually reduces variance; averaging
        # per-fold R2 values afterwards would not.
        ens = {}
        if "neural_2d" in preds and "neural_3d" in preds:
            ens["ens_2d3d"] = (preds["neural_2d"] + preds["neural_3d"]) / 2
        if "neural_2d" in preds and "traditional" in preds:
            ens["ens_neural_trad"] = (preds["neural_2d"] + preds["traditional"]) / 2
            ens["ens_all3"] = (preds["neural_2d"] + preds.get("neural_3d",
                               preds["neural_2d"]) + preds["traditional"]) / 3 \
                if "neural_3d" in preds else None
            ens = {k: v for k, v in ens.items() if v is not None}

        for name, pred in list(preds.items()) + list(ens.items()):
            per_fold.setdefault(name, []).extend(
                float(r2_score(y[f["test"]], pred[f["test"]])) for f in folds
            )

    if not per_fold:
        print("no completed runs found")
        return 2

    print(f"{'arm':16s} {'mean R2':>9} {'std':>8} {'n':>4}")
    for name, vals in sorted(per_fold.items(), key=lambda kv: -np.mean(kv[1])):
        v = np.asarray(vals)
        print(f"{name:16s} {v.mean():>9.4f} {v.std(ddof=1):>8.4f} {len(v):>4}")

    if corr:
        print("\nresidual correlation (lower = more complementary):")
        for k, v in sorted(corr.items()):
            print(f"  {k:28s} {np.mean(v):.4f}  {[round(x,4) for x in v]}")

    # paired verdicts against the strongest single arm
    singles = [k for k in per_fold if not k.startswith("ens_")]
    if singles:
        base = max(singles, key=lambda k: np.mean(per_fold[k]))
        print(f"\nbaseline = {base} ({np.mean(per_fold[base]):.4f})")
        for cand in sorted(per_fold):
            if not cand.startswith("ens_"):
                continue
            a, b = np.asarray(per_fold[cand]), np.asarray(per_fold[base])
            if len(a) != len(b):
                print(f"  {cand:16s} incomplete ({len(a)} folds vs {len(b)})")
                continue
            t, p = ttest_rel(a, b)
            d = a - b
            print(f"  {cand:16s} {a.mean():.4f}  delta {d.mean():+.4f}  "
                  f"t={t:5.2f} p={p:.4f}  wins {int((d>0).sum())}/{len(d)}")

    (ROOT / "5_outputs/ensemble_eval.json").write_text(json.dumps({
        "per_fold": {k: [round(v, 4) for v in vals] for k, vals in per_fold.items()},
        "means": {k: round(float(np.mean(v)), 4) for k, v in per_fold.items()},
        "residual_corr": {k: [round(x, 4) for x in v] for k, v in corr.items()},
    }, indent=2))
    print(f"\nwrote {ROOT / '5_outputs/ensemble_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
