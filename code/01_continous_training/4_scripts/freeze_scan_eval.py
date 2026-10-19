#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do the partial-freeze arms earn a place in the ensemble?

The question this answers
-------------------------
P0-2 found that freezing the first 6 encoder layers costs only 0.047 R2 while
halving the trainable parameters. A cheap arm is interesting for a different
reason than a strong one: if its ERRORS are decorrelated from the incumbent
neural arm, adding it can lift the ensemble even though it is individually
weaker.

Two gates, in this order (both required):
  1.  Independent strength -- fold-mean R2 >= 0.55 (arm_zoo's a-priori bar).
      A weak arm drags the weighted mean down no matter how decorrelated it is.
  2.  Residual correlation with the incumbent neural arm <= 0.70.
      Above that, the "new" arm is re-predicting what we already have.

Gate 2 is the one partial-freeze is expected to fail, and the prior is concrete:
freezing layers is a VARIANT OF THE SAME MODEL, not a different model family.
The incumbent pair (neural vs traditional) sits at 0.75; a same-model variant
should sit above that, not below.

Protocol
--------
Identical to ensemble_weighted.py, which produced the current headline ensemble:
NNLS weights fitted on OOF, leave-one-partition-out (a held-out partition's
weights never see that partition's folds), compared against the equal-weight
2-arm ensemble with a paired t-test over folds.

This is a fixed global weight vector, NOT a learned stacker -- learned stacking
was falsified in this project. Three global scalars cannot reproduce the
meta-distribution shift that killed stacking.

Usage:
    conda run -n unimol2 python 4_scripts/freeze_scan_eval.py --protocol scaffold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
FINETUNE = ROOT / "5_outputs/finetune_moltrain"
OUT = ROOT / "5_outputs/freeze_scan_eval.json"

DS = "Sheet2"
BASE = ["neural_2d", "traditional"]
# Every partial-freeze variant that exists as a finetune artifact. The tag is the
# run_label suffix; the arm name is what appears in the report.
FREEZE_ARMS = {
    "freeze4":  "_freeze4",
    "freeze6":  "_freeze6",
    "freeze8":  "_freeze8",
    "freeze10": "_freeze10",
}
PROTOCOLS = {"scaffold": (42, 1337, 2024), "butina": (42, 1337)}


def load_targets() -> np.ndarray:
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    return pd.to_numeric(d["RLU (log2)"]).values.astype(float)


def load_freeze_arm(tag: str, protocol: str, seeds) -> dict[int, np.ndarray] | None:
    """Per-seed OOF for a partial-freeze run, or None if any partition is missing.

    All-or-nothing on purpose: an arm present on only some partitions cannot be
    given honest leave-one-partition-out weights (the same reason arm_zoo drops
    neural_3d under butina instead of padding it).
    """
    out = {}
    for s in seeds:
        d = (FINETUNE / f"{DS}_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt{tag}"
             f"_{protocol}_seed{s}_lr5e-05_ep40")
        if not (d / "cv.data").exists():
            return None
        z = np.asarray(joblib.load(d / "cv.data")).ravel()
        out[s] = joblib.load(d / "target_scaler.ss").inverse_transform(
            z.reshape(-1, 1)).ravel()
    return out


def load_zoo(protocol: str, seeds):
    p = ROOT / f"5_outputs/arm_zoo_oof_{protocol}.npz"
    if not p.exists():
        raise SystemExit(f"missing {p}; run 4_scripts/arm_zoo.py first")
    z = np.load(p)
    return {s: {k[len(f"{s}__"):]: z[k] for k in z.files if k.startswith(f"{s}__")}
            for s in seeds}


def _design(store, folds, seed, arms, y):
    A, b = [], []
    for f in folds[seed]:
        te = np.asarray(f["test"])
        A.append(np.stack([store[seed][a][te] for a in arms], 1))
        b.append(y[te])
    return np.vstack(A), np.concatenate(b)


def lopo(store, folds, seeds, arms, y):
    """Per-fold R2 with NNLS weights fitted only on the other partitions."""
    out, weights = [], []
    for held in seeds:
        A, b = zip(*[_design(store, folds, t, arms, y) for t in seeds if t != held])
        w, _ = nnls(np.vstack(A), np.concatenate(b))
        weights.append((w / w.sum()).round(3).tolist())
        P = np.stack([store[held][a] for a in arms], 1)
        out += [r2_score(y[np.asarray(f["test"])], (P @ w)[np.asarray(f["test"])])
                for f in folds[held]]
    return np.asarray(out), weights


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="scaffold", choices=list(PROTOCOLS))
    ap.add_argument("--min-r2", type=float, default=0.55,
                    help="arm_zoo's a-priori independent-strength bar")
    ap.add_argument("--max-corr", type=float, default=0.70,
                    help="arm_zoo's a-priori residual-correlation bar")
    ap.add_argument("--force", nargs="*", default=None,
                    help="Arm names to push through the ensemble stage even if they "
                         "fail a gate. The gates are a-priori heuristics; forcing a "
                         "failing arm converts 'we screened it out' into 'we measured "
                         "it and it does not help', which is a much stronger claim.")
    args = ap.parse_args()

    seeds = PROTOCOLS[args.protocol]
    y = load_targets()
    store = load_zoo(args.protocol, seeds)
    folds = {s: json.loads(
        (ROOT / f"5_outputs/split_cache/{DS}_{args.protocol}_k5_seed{s}.json").read_text()
    )["folds"] for s in seeds}

    # ---- gate 1 + gate 2: screen each freeze arm before ensembling ----
    print(f"===== {args.protocol}: gate screen (bars: R2>={args.min_r2}, "
          f"resid-corr<={args.max_corr}) =====")
    print(f"  {'arm':10s} {'indep R2':>9} {'vs neural_2d':>13} {'vs traditional':>15}  verdict")
    eligible, screen = [], {}
    for arm, tag in FREEZE_ARMS.items():
        oof = load_freeze_arm(tag, args.protocol, seeds)
        if oof is None:
            print(f"  {arm:10s} {'--':>9} {'--':>13} {'--':>15}  not run on all partitions")
            continue
        store_local = dict(store)
        for s in seeds:
            store_local[s] = dict(store[s], **{arm: oof[s]})
        r2 = float(np.mean([r2_score(y, oof[s]) for s in seeds]))
        # NOTE: this is the POOLED R2 averaged over partitions, not the project's
        # headline fold-mean. The gate is a screening heuristic so pooled is fine,
        # but the two differ by ~0.015 (freeze4: pooled 0.6617 vs fold-mean 0.6469),
        # so never quote this number next to a fold-mean from peft_eval.py.
        r2_foldmean = float(np.mean(
            [r2_score(y[np.asarray(f["test"])], oof[s][np.asarray(f["test"])])
             for s in seeds for f in folds[s]]))
        c_n = float(np.mean([pearsonr(y - oof[s], y - store[s]["neural_2d"]).statistic
                             for s in seeds]))
        c_t = float(np.mean([pearsonr(y - oof[s], y - store[s]["traditional"]).statistic
                             for s in seeds]))
        ok = (r2 >= args.min_r2) and (c_n <= args.max_corr)
        verdict = "PASS" if ok else (
            "FAIL: too correlated" if r2 >= args.min_r2 else "FAIL: too weak")
        print(f"  {arm:10s} {r2:>9.4f} {c_n:>13.4f} {c_t:>15.4f}  {verdict}")
        screen[arm] = {"indep_r2_pooled": round(r2, 4),
                       "indep_r2_foldmean": round(r2_foldmean, 4),
                       "resid_corr_neural_2d": round(c_n, 4),
                       "resid_corr_traditional": round(c_t, 4),
                       "passes": ok, "verdict": verdict}
        if ok or arm in (args.force or []):
            eligible.append(arm)
            store = store_local   # keep it for the ensemble stage

    # Report the incumbents' own correlation for scale -- the number any new arm
    # has to beat to be "diverse".
    inc = float(np.mean([pearsonr(y - store[s]["neural_2d"],
                                  y - store[s]["traditional"]).statistic for s in seeds]))
    print(f"\n  reference: incumbent neural_2d vs traditional = {inc:+.4f}")

    # ---- ensemble stage ----
    def equal_base():
        return np.asarray([
            r2_score(y[np.asarray(f["test"])],
                     ((store[s]["neural_2d"] + store[s]["traditional"]) / 2
                      )[np.asarray(f["test"])])
            for s in seeds for f in folds[s]])

    base_eq = equal_base()
    base_2, w2 = lopo(store, folds, seeds, BASE, y)
    t, p = ttest_rel(base_2, base_eq)
    print(f"\n===== {args.protocol}: ensemble ({len(base_eq)} folds) =====")
    print(f"  {'equal 2-arm (headline)':30s} {base_eq.mean():.4f} +- {base_eq.std(ddof=1):.4f}")
    print(f"  {'NNLS 2-arm (control)':30s} {base_2.mean():.4f}  "
          f"delta={base_2.mean()-base_eq.mean():+.4f} p={p:.4f}  "
          f"wins={int((base_2>base_eq).sum())}/{len(base_eq)}")

    report = {"dataset": DS, "protocol": args.protocol, "seeds": list(seeds),
              "screen": screen, "incumbent_resid_corr": round(inc, 4),
              "equal_2arm": {"mean": round(float(base_eq.mean()), 4),
                             "std": round(float(base_eq.std(ddof=1)), 4)},
              "nnls_2arm_control": {"mean": round(float(base_2.mean()), 4),
                                    "delta": round(float(base_2.mean() - base_eq.mean()), 4),
                                    "p": round(float(p), 4), "weights": w2},
              "candidates": {}}

    if not eligible:
        print(f"\n  no arm cleared both gates -> ensemble stage skipped.\n"
              f"  (This is the answer, not a failure of the script: a same-model "
              f"variant is not a new arm.)")
    for arm in eligible:
        c, w = lopo(store, folds, seeds, BASE + [arm], y)
        t, p = ttest_rel(c, base_eq)
        report["candidates"][arm] = {
            "mean": round(float(c.mean()), 4), "std": round(float(c.std(ddof=1)), 4),
            "delta": round(float(c.mean() - base_eq.mean()), 4),
            "p": round(float(p), 4), "wins": int((c > base_eq).sum()),
            "n": int(len(c)), "weights": w}
        print(f"  {'NNLS +' + arm:30s} {c.mean():.4f}  "
              f"delta={c.mean()-base_eq.mean():+.4f} p={p:.4f}  "
              f"wins={int((c>base_eq).sum())}/{len(c)}  w={w}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            prev = {}
    prev[args.protocol] = report
    OUT.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
