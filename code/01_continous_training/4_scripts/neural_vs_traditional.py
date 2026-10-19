#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final verdict: neural fine-tune vs the TUNED traditional baseline, paired by fold.

Both sides run the same protocol (finetune_moltrain's cached scaffold folds, fold-mean
R2 on log2 RLU) over the same 3 partitions (seeds 42 / 1337 / 2024) = 15 folds.

Why paired: the two models see identical train/test splits, so the fold is the natural
matched unit. Comparing two independent means would drown a real effect in the large
between-partition variance that both models share.

Why 15 folds and not 5: a single partition's fold-mean carries ~0.05 of split-assignment
noise (measured earlier), which is larger than most effects under study. Repeating the
partition is what makes a 0.05 gap resolvable at all.

Neural numbers come from each run's cv.data (OOF predictions, stored z-scored) plus its
target_scaler.ss. Traditional numbers come from traditional_baseline_tuned.json, which
stores per-fold R2 only -- so the paired test is done on per-fold R2, the unit both
sides share.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
FMT = "Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed{}_lr5e-05_ep40"
SEEDS = (42, 1337, 2024)
DS = "Sheet2"


def neural_per_fold(seed: int):
    """Per-fold R2 for the v5 fine-tune on one scaffold partition.

    cv.data holds z-scored OOF predictions (MolTrain stores cv['pred'] before the
    inverse_transform), so the scaler must be applied before scoring. Getting this
    wrong leaves R2 looking like -0.8 with healthy rank metrics -- the failure mode
    documented in finetune_moltrain.py.
    """
    p = ROOT / "5_outputs/finetune_moltrain" / FMT.format(seed)
    if not (p / "cv.data").exists():
        return None
    pred_z = np.asarray(joblib.load(p / "cv.data")).ravel()
    scaler = joblib.load(p / "target_scaler.ss")
    pred = scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()

    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    folds = json.loads(
        (ROOT / f"5_outputs/split_cache/{DS}_scaffold_k5_seed{seed}.json").read_text()
    )["folds"]
    from sklearn.metrics import r2_score
    return [float(r2_score(y[f["test"]], pred[f["test"]])) for f in folds]


def main():
    trad = json.loads((ROOT / "5_outputs/traditional_baseline_tuned.json").read_text())
    best_key = max(trad["results"], key=lambda k: trad["results"][k]["r2_mean_15"])
    best = trad["results"][best_key]
    print(f"best traditional: {best_key}")
    print(f"  {best['r2_mean_15']:.4f} +- {best['r2_std_15']:.4f} over {best['n_folds']} folds")

    print(f"\nneural: v5 fine-tune, same folds")
    neu_all, trad_all, rows = [], [], []
    for seed in SEEDS:
        pf = neural_per_fold(seed)
        if pf is None:
            print(f"  seed {seed}: NOT YET RUN -> {FMT.format(seed)}")
            continue
        tpf = best["per_seed"][str(seed)]["folds"]
        neu_all.extend(pf)
        trad_all.extend(tpf)
        rows.append((seed, float(np.mean(pf)), float(np.mean(tpf))))
        print(f"  seed {seed}: neural {np.mean(pf):.4f}   traditional {np.mean(tpf):.4f}")

    if len(neu_all) < 5:
        print("\nnot enough folds to compare yet")
        return 2

    neu_all = np.asarray(neu_all)
    trad_all = np.asarray(trad_all)
    print(f"\n=== {len(neu_all)} paired folds ===")
    print(f"  neural      {neu_all.mean():.4f} +- {neu_all.std(ddof=1):.4f}")
    print(f"  traditional {trad_all.mean():.4f} +- {trad_all.std(ddof=1):.4f}")
    diff = neu_all - trad_all
    t, p = ttest_rel(neu_all, trad_all)
    print(f"\npaired t-test (neural - traditional):")
    print(f"  mean delta = {diff.mean():+.4f}   t = {t:.3f}   p = {p:.4f}")
    print(f"  neural wins {int((diff > 0).sum())}/{len(diff)} folds")

    if p < 0.05 and diff.mean() > 0:
        verdict = ("NEURAL WINS: the 3D fine-tuned model beats the tuned traditional "
                   "baseline significantly. The 'traditional is stronger' premise is "
                   "rejected -> no case for distilling it into the neural model.")
    elif p < 0.05 and diff.mean() < 0:
        verdict = ("TRADITIONAL WINS: the premise holds -> the fusion route is "
                   "justified. Distil (auxiliary-loss), do not concat raw fingerprints.")
    else:
        verdict = ("NO SIGNIFICANT DIFFERENCE: the two are within noise of each other "
                   "over 15 folds. Neither dominates, which makes the two-arm ENSEMBLE "
                   "the more promising route than making either one stronger alone.")
    print(f"\nVERDICT: {verdict}")

    (ROOT / "5_outputs/neural_vs_traditional.json").write_text(json.dumps({
        "neural": "v5 fine-tune", "traditional": best_key,
        "folds": len(neu_all),
        "neural_fold_r2": [round(float(v), 4) for v in neu_all],
        "traditional_fold_r2": [round(float(v), 4) for v in trad_all],
        "mean_neural": round(float(neu_all.mean()), 4),
        "mean_traditional": round(float(trad_all.mean()), 4),
        "delta": round(float(diff.mean()), 4),
        "t": round(float(t), 4), "p": round(float(p), 4),
        "per_seed": [{"seed": s, "neural": n, "traditional": tr} for s, n, tr in rows],
        "verdict": verdict,
    }, indent=2))
    print(f"\nwrote {ROOT / '5_outputs/neural_vs_traditional.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
