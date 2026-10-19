#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired comparison of fine-tune arms: PEFT (adapter / frozen) vs the full fine-tune.

The same convention as neural_vs_traditional.py, generalised: OOF predictions are read
from each run's own artifact directory, un-standardised with that run's target_scaler.ss,
scored per fold against the cached folds, and compared with a PAIRED t-test over folds.

Why paired, why repeated partitions: the fold is the natural matched unit (both arms see
identical train/test rows), and a single 5-fold partition carries ~0.025 of run-to-run
noise -- the same size as the effects under study here. 3 partitions x 5 folds = 15
paired folds is what makes a 0.01-0.02 gap resolvable at all.

Artifact naming is the interface, and it comes from finetune_moltrain.build_run_label:

    Sheet2_exp_<run_label>_<split>_seed<seed>_lr<lr>_ep<epochs>[_3dconf]/

where run_label ends in `_frozen` (frozen backbone) and/or `_adapter<r>`:
    A (full fine-tune)     unimol2_lipids_84m_v5/checkpoint_1_30000.pt
    B (frozen + linear)    ...checkpoint_1_30000.pt_frozen
    C (frozen + adapter)   ...checkpoint_1_30000.pt_frozen_adapter48
    D (freeze6 + adapter)  ...checkpoint_1_30000.pt_freeze6_adapter48

Usage:
    conda run -n unimol2 python 4_scripts/peft_eval.py --split scaffold
    conda run -n unimol2 python 4_scripts/peft_eval.py --split butina --seeds 42 1337 2024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "5_outputs/finetune_moltrain"
SPLIT_CACHE = REPO_ROOT / "5_outputs/split_cache"
OUT_JSON = REPO_ROOT / "5_outputs/peft_eval.json"

# The checkpoint stem every v5 arm shares. Two spellings because finetune_moltrain
# uses the raw run_label (`variant/checkpoint`) as the aggregate-results key but
# `run_label.replace('/', '_')` in the output DIRECTORY name. Keeping both here makes
# the mismatch explicit rather than letting one silently fail to match.
V5_LABEL = "unimol2_lipids_84m_v5/checkpoint_1_30000.pt"
V5_STEM = V5_LABEL.replace("/", "_")

# Arm definitions. `suffix` is appended to V5_STEM to form the run_label; the baseline
# (A) is the bare stem. `expect` is the trainable-parameter count the run should have
# reported, used to catch an arm that silently ran as something else.
ARMS = {
    "A_full":     {"suffix": "",                    "expect_params": 83_426_289,
                   "desc": "full fine-tune (baseline)"},
    "B_frozen":   {"suffix": "_frozen",             "expect_params": 769,
                   "desc": "frozen backbone + linear head"},
    "C_adapter":  {"suffix": "_frozen_adapter48",   "expect_params": 913_729,
                   "desc": "frozen backbone + bottleneck adapter r=48 + head"},
    # D vs D0 isolates what the adapters contribute once the backbone already trains:
    # identical freeze, same data, same folds, adapters the only difference.
    "D0_f6":      {"suffix": "_freeze6",            "expect_params": None,
                   "desc": "frozen first 6 layers, NO adapter (control)"},
    "D_f6adapt":  {"suffix": "_freeze6_adapter48",  "expect_params": 44_011_281,
                   "desc": "frozen first 6 layers + adapter r=48"},
    # Layer scan (run_freeze_scan.sh): find where the partial-freeze sweet spot is.
    # These are the same `--freeze-layers N` mechanism as D0, no adapters.
    "F4":         {"suffix": "_freeze4",            "expect_params": None,
                   "desc": "frozen first 4 layers"},
    "F8":         {"suffix": "_freeze8",            "expect_params": None,
                   "desc": "frozen first 8 layers"},
    "F10":        {"suffix": "_freeze10",           "expect_params": None,
                   "desc": "frozen first 10 layers"},
}


def load_targets(dataset: str) -> np.ndarray:
    import pandas as pd
    xlsx = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
    df = pd.read_excel(xlsx, sheet_name=dataset)
    col = "RLU (log2)" if dataset == "Sheet2" else "IGROV1_transfection_efficiency"
    y = df[col].astype(float).to_numpy()
    if dataset != "Sheet2":
        y = np.log10(y)
    return y


def run_dir(arm: str, dataset: str, split: str, seed: int, lr: float,
            epochs: int) -> Path:
    label = f"{V5_STEM}{ARMS[arm]['suffix']}"
    return OUT_DIR / (f"{dataset}_exp_{label}_{split}_seed{seed}"
                      f"_lr{lr}_ep{epochs}")


def arm_per_fold(arm: str, dataset: str, split: str, seed: int, lr: float,
                 epochs: int, kfold: int) -> list[float] | None:
    """Per-fold R2 for one arm on one partition, or None if it has not been run.

    cv.data holds z-scored OOF predictions, so the run's own target_scaler.ss must be
    applied before scoring (坑 #4) -- skipping that leaves R2 near -0.8 with healthy
    rank metrics.
    """
    d = run_dir(arm, dataset, split, seed, lr, epochs)
    if not (d / "cv.data").exists():
        return None
    pred = joblib.load(d / "target_scaler.ss").inverse_transform(
        np.asarray(joblib.load(d / "cv.data")).reshape(-1, 1)).ravel()
    y = load_targets(dataset)
    if len(pred) != len(y):
        raise SystemExit(f"{d.name}: {len(pred)} predictions vs {len(y)} targets")
    folds = json.loads(
        (SPLIT_CACHE / f"{dataset}_{split}_k{kfold}_seed{seed}.json").read_text()
    )["folds"]
    return [float(r2_score(y[f["test"]], pred[f["test"]])) for f in folds]


def arms_present() -> None:
    """Report the architecture each arm actually built, and its trainable count.

    Two checks that catch the failure mode this project keeps hitting (坑 #6): an arm
    whose flag never reached the model still produces plausible artifacts.

    1. `adapters.*` keys in a saved state_dict -- present iff --adapter-dim reached the
       model. A saved state_dict keeps frozen tensors, so it cannot tell what was
       trainable, but it does reveal the architecture.
    2. The trainable count recorded at run time in moltrain_results.json. That number
       is NOT recoverable after the fact, which is why finetune_moltrain.py stores it.
    """
    import torch
    agg = {}
    agg_path = OUT_DIR / "moltrain_results.json"
    if agg_path.exists():
        agg = json.loads(agg_path.read_text()).get("runs", {})

    def recorded_params(arm: str):
        """Trainable count this arm's run recorded, or None.

        Aggregate keys are `<run_label>|<ds>|...` and run_label is exactly
        V5_STEM + suffix, so the checkpoint field must be compared for EQUALITY. A
        substring test would let B ("_frozen") match C's key ("_frozen_adapter48") and
        silently report C's number as B's.
        """
        want = f"{V5_LABEL}{ARMS[arm]['suffix']}"
        for k, v in agg.items():
            if k.split("|")[0] != want:
                continue
            n = v.get("metrics", {}).get("Sheet2", {}).get("trainable_params")
            if n is not None:
                return n
        return None

    print("\narm integrity:")
    for arm, meta in ARMS.items():
        has_adapter = None
        for seed in (42, 1337, 2024):
            f = run_dir(arm, "Sheet2", "scaffold", seed, 5e-5, 40) / "model_0.pth"
            if not f.exists():
                continue
            sd = torch.load(f, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            has_adapter = any(k.startswith("adapters.") for k in sd)
            break
        want_adapter = "adapter" in meta["suffix"]
        recorded = recorded_params(arm)
        flag = ""
        if has_adapter is not None and has_adapter != want_adapter:
            flag = "  <-- MISMATCH: adapter flag did not reach the model"
        elif recorded is not None and meta["expect_params"] is not None \
                and recorded != meta["expect_params"]:
            flag = (f"  <-- MISMATCH: trained {recorded:,} params, "
                    f"expected {meta['expect_params']:,}")
        state = ("adapter keys present" if has_adapter
                 else "no adapter keys" if has_adapter is False else "not run")
        rec = f"  trainable={recorded:,}" if recorded is not None else ""
        print(f"  {arm:11s} {state:20s}{rec}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Sheet2")
    ap.add_argument("--split", default="scaffold", choices=["scaffold", "butina"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--baseline", default="A_full",
                    help="arm every other arm is paired against")
    args = ap.parse_args()

    print(f"{args.dataset}  split={args.split}  seeds={args.seeds}  "
          f"baseline={args.baseline}")
    arms_present()
    print()

    per_arm: dict[str, list[float]] = {}
    per_arm_seed: dict[str, list[tuple[int, float]]] = {}
    std_by_arm: dict[str, list[float]] = {}
    for arm, meta in ARMS.items():
        fold_r2, seed_means, within_seed_std = [], [], []
        for seed in args.seeds:
            pf = arm_per_fold(arm, args.dataset, args.split, seed, args.lr,
                              args.epochs, args.kfold)
            if pf is None:
                continue
            fold_r2.extend(pf)
            seed_means.append((seed, float(np.mean(pf))))
            within_seed_std.append(float(np.std(pf, ddof=1)))
        if not fold_r2:
            print(f"{arm:11s} {meta['desc']:52s} NOT RUN")
            continue
        per_arm[arm] = fold_r2
        per_arm_seed[arm] = seed_means
        std_by_arm[arm] = within_seed_std
        print(f"{arm:11s} {meta['desc']:52s} n={len(fold_r2):2d} "
              f"fold-mean R2 = {np.mean(fold_r2):.4f} "
              f"(within-partition std {np.mean(within_seed_std):.4f})")

    base = args.baseline
    if base not in per_arm:
        print(f"\nbaseline {base} has no artifacts yet; nothing to pair against")
        return 2

    print(f"\n=== paired vs {base} ({len(per_arm[base])} folds) ===")
    print(f"{'arm':11s} {'mean':>8} {'delta':>8} {'t':>8} {'p':>8}  wins")
    rows = {}
    for arm, vals in per_arm.items():
        if arm == base:
            continue
        if len(vals) != len(per_arm[base]):
            print(f"{arm:11s} fold count {len(vals)} != baseline "
                  f"{len(per_arm[base])} -- skipped (must pair fold-for-fold)")
            continue
        a, b = np.asarray(vals), np.asarray(per_arm[base])
        diff = a - b
        t, p = ttest_rel(a, b)
        wins = int((diff > 0).sum())
        print(f"{arm:11s} {a.mean():>8.4f} {diff.mean():>+8.4f} {t:>8.3f} {p:>8.4f}  "
              f"{wins}/{len(diff)}")
        rows[arm] = {
            "desc": ARMS[arm]["desc"],
            "mean": round(float(a.mean()), 4),
            "delta": round(float(diff.mean()), 4),
            "t": round(float(t), 4), "p": round(float(p), 4),
            "wins": wins, "n_folds": len(diff),
            # PEFT's selling point is the variance, not only the mean: report the
            # within-partition fold spread for each arm side by side.
            "mean_within_seed_std": round(float(np.mean(std_by_arm[arm])), 4),
            "baseline_within_seed_std": round(float(np.mean(std_by_arm[base])), 4),
            "fold_r2": [round(float(v), 4) for v in vals],
            "per_seed": [{"seed": s, "mean": round(m, 4)} for s, m in per_arm_seed[arm]],
        }

    # The decisive contrast for this project's PEFT question: does the adapter still
    # contribute once the backbone is already partly trainable? D0 and D differ ONLY by
    # the adapter (same freeze list, same folds, same data), which isolates the adapter's
    # marginal value cleanly -- unlike D-vs-A, which mixes it with the cost of freezing.
    out_extra = {}
    if "D0_f6" in per_arm and "D_f6adapt" in per_arm \
            and len(per_arm["D0_f6"]) == len(per_arm["D_f6adapt"]):
        d0 = np.asarray(per_arm["D0_f6"])
        d1 = np.asarray(per_arm["D_f6adapt"])
        t, p = ttest_rel(d1, d0)
        print(f"\n=== adapter's MARGINAL value under freeze6 (D - D0, {len(d0)} folds) ===")
        print(f"  D0 (freeze6, no adapter) {d0.mean():.4f}")
        print(f"  D  (freeze6 + adapter)   {d1.mean():.4f}")
        print(f"  adapter contribution     {(d1-d0).mean():+.4f}  t={t:.3f}  p={p:.4f}  "
              f"D wins {int((d1 > d0).sum())}/{len(d0)}")
        out_extra["adapter_marginal_under_freeze6"] = {
            "mean_d0": round(float(d0.mean()), 4),
            "mean_d": round(float(d1.mean()), 4),
            "delta": round(float((d1 - d0).mean()), 4),
            "t": round(float(t), 4), "p": round(float(p), 4),
            "wins": int((d1 > d0).sum()), "n_folds": len(d0),
        }

    out = {
        "dataset": args.dataset, "split": args.split, "seeds": args.seeds,
        "baseline": base, "baseline_mean": round(float(np.mean(per_arm[base])), 4),
        "folds": len(per_arm[base]),
        "arms": rows,
        **out_extra,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text())
        except json.JSONDecodeError:
            prev = {}
    prev[f"{args.dataset}|{args.split}|seeds{'-'.join(map(str,args.seeds))}"] = out
    OUT_JSON.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
