#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensemble evaluation on frozen 25-fold downstream grids.

Context & Motivation
--------------------
Ensembling the neural model (LipidFM v5) with classical descriptor models
(e.g., tuned_gbm) is a proven lever for variance reduction and complementary signal
capture (residual correlation ~0.75).

This script performs rigorous out-of-fold (OOF) ensembling on the frozen 25-fold
grid (5 partition seeds x 5 folds = 25 folds):
1. Reads existing per-molecule OOF predictions from ``5_outputs/downstream_predictions/``.
2. Evaluates both EQUAL weighting and Leave-One-Partition-Out (LOPO) NNLS weighting:
   - For LOPO NNLS: weights for a held-out seed partition are fitted ONLY on the
     OOF predictions of the other 4 seed partitions. No test fold or seed partition
     ever sees its own weights.
3. Computes paired Wilcoxon signed-rank tests against:
   - Base single models (v5, v5_esval, tuned_gbm)
   - External benchmark model (lumi_lab_esval)
4. Reports both raw 25-fold slot metrics and de-duplicated metrics (坑 #17: test set
   identity dedup via frozenset(test_idx), crucial for Sheet3/butina which only
   has 11 distinct test sets).
5. Saves clean structured JSON output to ``5_outputs/ensemble_frozen_eval.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
PRED_DIR = ROOT / "5_outputs/downstream_predictions"
SPLIT_DIR = ROOT / "5_outputs/downstream_splits"
OUT_JSON = ROOT / "5_outputs/ensemble_frozen_eval.json"

SEEDS = [42, 123, 456, 789, 1011]
COMBOS = [("Sheet2", "scaffold"), ("Sheet2", "butina"), ("Sheet3", "butina")]


def load_split_folds(dataset: str, split: str, seed: int) -> List[Dict[str, Any]]:
    path = SPLIT_DIR / f"{dataset}_{split}_k5_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["folds"]


def load_oof(dataset: str, arm: str, split: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    path = PRED_DIR / f"{dataset}|{arm}|{split}|seed{seed}|k5.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file missing: {path}")
    data = np.load(path)
    return np.asarray(data["oof"], float).ravel(), np.asarray(data["y"], float).ravel()


def eval_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": r2, "rmse": rmse}


def compute_lopo_nnls_weights(
    train_seeds: Sequence[int],
    arms: Sequence[str],
    oof_store: Dict[int, Dict[str, np.ndarray]],
    y_store: Dict[int, np.ndarray],
    folds_store: Dict[int, List[Dict[str, Any]]],
) -> np.ndarray:
    A_blocks = []
    b_blocks = []
    for s in train_seeds:
        for fold in folds_store[s]:
            te_idx = fold["test_idx"]
            cols = [oof_store[s][a][te_idx] for a in arms]
            A_blocks.append(np.stack(cols, axis=1))
            b_blocks.append(y_store[s][te_idx])
    A = np.vstack(A_blocks)
    b = np.concatenate(b_blocks)
    w, _ = nnls(A, b)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.full(len(arms), 1.0 / len(arms))
    return w


def paired_stats(arr_ref: np.ndarray, arr_comp: np.ndarray) -> Dict[str, Any]:
    diff = arr_ref - arr_comp
    w_res = wilcoxon(diff)
    return {
        "mean_diff": float(np.mean(diff)),
        "p_value": float(w_res.pvalue),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
        "n": len(diff),
    }


def analyze_combo(dataset: str, split: str) -> Dict[str, Any]:
    folds_store = {s: load_split_folds(dataset, split, s) for s in SEEDS}

    # Available candidate arms to consider
    candidate_arms = [
        "lipidfm_v5",
        "lipidfm_v5_esval",
        "tuned_gbm",
        "mlp_morgan",
        "ecfp4_xgb",
        "ecfp4_rf",
        "xgb_atompair",
        "lumi_lab_esval",
        "stock_84m",
    ]
    available_arms = []
    oof_store: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in SEEDS}
    y_store: Dict[int, np.ndarray] = {}

    for arm in candidate_arms:
        if all((PRED_DIR / f"{dataset}|{arm}|{split}|seed{s}|k5.npz").exists() for s in SEEDS):
            available_arms.append(arm)
            for s in SEEDS:
                oof, y = load_oof(dataset, arm, split, s)
                oof_store[s][arm] = oof
                y_store[s] = y

    # Single arm performance
    single_arm_metrics = {}
    for arm in available_arms:
        r2_list = []
        for s in SEEDS:
            for fold in folds_store[s]:
                te_idx = fold["test_idx"]
                m = eval_fold_metrics(y_store[s][te_idx], oof_store[s][arm][te_idx])
                r2_list.append(m["r2"])
        single_arm_metrics[arm] = {
            "r2_per_fold": r2_list,
            "mean_r2": float(np.mean(r2_list)),
            "std_r2": float(np.std(r2_list, ddof=1)),
        }

    # Ensemble configs to test
    ensemble_specs = [
        {"name": "ens2_default_eq", "arms": ["lipidfm_v5", "tuned_gbm"], "method": "equal"},
        {"name": "ens2_default_lopo", "arms": ["lipidfm_v5", "tuned_gbm"], "method": "lopo_nnls"},
        {"name": "ens2_esval_eq", "arms": ["lipidfm_v5_esval", "tuned_gbm"], "method": "equal"},
        {"name": "ens2_esval_lopo", "arms": ["lipidfm_v5_esval", "tuned_gbm"], "method": "lopo_nnls"},
        {"name": "ens3_atompair_default_eq", "arms": ["lipidfm_v5", "tuned_gbm", "xgb_atompair"], "method": "equal"},
        {"name": "ens3_atompair_default_lopo", "arms": ["lipidfm_v5", "tuned_gbm", "xgb_atompair"], "method": "lopo_nnls"},
        {"name": "ens3_atompair_esval_eq", "arms": ["lipidfm_v5_esval", "tuned_gbm", "xgb_atompair"], "method": "equal"},
        {"name": "ens3_atompair_esval_lopo", "arms": ["lipidfm_v5_esval", "tuned_gbm", "xgb_atompair"], "method": "lopo_nnls"},
        {"name": "ens3_mlp_esval_eq", "arms": ["lipidfm_v5_esval", "tuned_gbm", "mlp_morgan"], "method": "equal"},
        {"name": "ens3_mlp_esval_lopo", "arms": ["lipidfm_v5_esval", "tuned_gbm", "mlp_morgan"], "method": "lopo_nnls"},
        {"name": "ens4_all_esval_lopo", "arms": ["lipidfm_v5_esval", "tuned_gbm", "xgb_atompair", "mlp_morgan"], "method": "lopo_nnls"},
    ]

    ensemble_results = {}
    for spec in ensemble_specs:
        ens_name = spec["name"]
        arms = spec["arms"]
        method = spec["method"]

        if not all(a in available_arms for a in arms):
            continue

        r2_per_fold = []
        weights_by_seed = {}
        ens_oof_by_seed: Dict[int, np.ndarray] = {}

        if method == "equal":
            w = np.full(len(arms), 1.0 / len(arms))
            for s in SEEDS:
                weights_by_seed[s] = w.tolist()
                pred = sum(w[i] * oof_store[s][a] for i, a in enumerate(arms))
                ens_oof_by_seed[s] = pred
                for fold in folds_store[s]:
                    te_idx = fold["test_idx"]
                    m = eval_fold_metrics(y_store[s][te_idx], pred[te_idx])
                    r2_per_fold.append(m["r2"])
        elif method == "lopo_nnls":
            for held_seed in SEEDS:
                train_seeds = [s for s in SEEDS if s != held_seed]
                w = compute_lopo_nnls_weights(train_seeds, arms, oof_store, y_store, folds_store)
                weights_by_seed[held_seed] = w.round(4).tolist()
                pred = sum(w[i] * oof_store[held_seed][a] for i, a in enumerate(arms))
                ens_oof_by_seed[held_seed] = pred
                for fold in folds_store[held_seed]:
                    te_idx = fold["test_idx"]
                    m = eval_fold_metrics(y_store[held_seed][te_idx], pred[te_idx])
                    r2_per_fold.append(m["r2"])

        # Compare vs baseline references
        r2_arr = np.array(r2_per_fold)
        comp_dict = {}
        for ref_arm in ["lipidfm_v5", "lipidfm_v5_esval", "tuned_gbm", "lumi_lab_esval"]:
            if ref_arm in single_arm_metrics:
                ref_arr = np.array(single_arm_metrics[ref_arm]["r2_per_fold"])
                comp_dict[f"vs_{ref_arm}"] = paired_stats(r2_arr, ref_arr)

        # Dedup analysis (坑 #17)
        seen_test_sets: Dict[frozenset, List[Tuple[int, List[int]]]] = {}
        for s in SEEDS:
            for fold in folds_store[s]:
                te_key = frozenset(fold["test_idx"])
                seen_test_sets.setdefault(te_key, []).append((s, fold["test_idx"]))

        dedup_first_r2 = []
        dedup_mean_r2 = []
        lumi_first_r2 = []
        lumi_mean_r2 = []
        has_lumi = "lumi_lab_esval" in available_arms

        for te_key, occurrences in seen_test_sets.items():
            # First seed occurrence
            s0, idx0 = occurrences[0]
            y0 = y_store[s0][idx0]
            dedup_first_r2.append(eval_fold_metrics(y0, ens_oof_by_seed[s0][idx0])["r2"])
            if has_lumi:
                lumi_first_r2.append(eval_fold_metrics(y0, oof_store[s0]["lumi_lab_esval"][idx0])["r2"])

            # Mean across seed occurrences
            sub_ens = [eval_fold_metrics(y_store[si][idxi], ens_oof_by_seed[si][idxi])["r2"] for si, idxi in occurrences]
            dedup_mean_r2.append(float(np.mean(sub_ens)))
            if has_lumi:
                sub_lumi = [eval_fold_metrics(y_store[si][idxi], oof_store[si]["lumi_lab_esval"][idxi])["r2"] for si, idxi in occurrences]
                lumi_mean_r2.append(float(np.mean(sub_lumi)))

        dedup_analysis = {
            "n_distinct_test_sets": len(seen_test_sets),
            "first_seed_mean_r2": float(np.mean(dedup_first_r2)),
            "seed_mean_r2": float(np.mean(dedup_mean_r2)),
        }
        if has_lumi:
            dedup_analysis["vs_lumi_first_seed"] = paired_stats(np.array(dedup_first_r2), np.array(lumi_first_r2))
            dedup_analysis["vs_lumi_seed_mean"] = paired_stats(np.array(dedup_mean_r2), np.array(lumi_mean_r2))

        ensemble_results[ens_name] = {
            "arms": arms,
            "method": method,
            "mean_r2": float(np.mean(r2_per_fold)),
            "std_r2": float(np.std(r2_per_fold, ddof=1)),
            "weights": weights_by_seed,
            "comparisons": comp_dict,
            "dedup_analysis": dedup_analysis,
        }

    return {
        "dataset": dataset,
        "split": split,
        "single_models": {k: {"mean_r2": v["mean_r2"], "std_r2": v["std_r2"]} for k, v in single_arm_metrics.items()},
        "ensembles": ensemble_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensembles on frozen 25-fold grids.")
    parser.add_argument("--save", action="store_true", default=True, help="Save summary to JSON")
    args = parser.parse_args()

    all_results = {}
    print("=" * 80)
    print("ENFORCED FROZEN 25-FOLD ENSEMBLE EVALUATION (TE)")
    print("=" * 80)

    for ds, sp in COMBOS:
        print(f"\n>>> Running evaluation on: {ds} / {sp}")
        res = analyze_combo(ds, sp)
        all_results[f"{ds}_{sp}"] = res

        print("  Single Models:")
        for m_name, m_data in sorted(res["single_models"].items()):
            print(f"    {m_name:20s}: R² = {m_data['mean_r2']:+.4f} ± {m_data['std_r2']:.4f}")

        print("  Ensembles:")
        for ens_name, ens_data in res["ensembles"].items():
            print(f"\n    [{ens_name}] ({ens_data['method']})")
            print(f"      Arms: {', '.join(ens_data['arms'])}")
            print(f"      25-fold R² = {ens_data['mean_r2']:+.4f} ± {ens_data['std_r2']:.4f}")
            for comp_name, comp_stat in ens_data["comparisons"].items():
                print(
                    f"        {comp_name:18s}: Δ = {comp_stat['mean_diff']:+.4f}, "
                    f"p = {comp_stat['p_value']:.4g}, W/L/T = {comp_stat['wins']}/{comp_stat['losses']}/{comp_stat['ties']}"
                )
            d = ens_data["dedup_analysis"]
            print(f"      坑 #17 Dedup ({d['n_distinct_test_sets']} distinct test sets):")
            print(f"        first_seed R² = {d['first_seed_mean_r2']:+.4f}, seed_mean R² = {d['seed_mean_r2']:+.4f}")
            if "vs_lumi_seed_mean" in d:
                c = d["vs_lumi_seed_mean"]
                print(f"        vs LUMI (seed_mean): Δ = {c['mean_diff']:+.4f}, p = {c['p_value']:.4g}, wins = {c['wins']}/{c['n']}")

    if args.save:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[OK] Results saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
