#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weighted third arm: does adding a decorrelated model help BOTH protocols?

Context
-------
arm_zoo.py screened ten new arms. None cleared the a-priori bar for
EQUAL-weight addition, and equal-weight addition actively hurt
(scaffold -0.031, butina -0.042) -- because every candidate is individually
weaker than the two incumbents, so averaging drags the prediction toward the
weak arm. That is an artefact of the WEIGHTING, not evidence against diversity.

This script tests the question the right way: fit non-negative weights on the
out-of-fold predictions, then add the candidate with those weights.

Validation design
-----------------
Leave-one-partition-out. Weights for a held-out partition are fitted ONLY on
the other partitions' OOF predictions, so no fold ever sees its own weights.
Comparing against the equal-weight 2-arm ensemble (the current headline,
0.7037 / 0.8062) keeps the baseline identical to 11-最终结论汇总.md.

The control that matters: NNLS-weighting the TWO INCUMBENT arms alone is also
measured. Without it, any gain could just be "weighting beats averaging" rather
than "the new arm carries signal". The control settles that.

Note on scope: this is a blend of already-computed OOF, i.e. a fixed global
weight vector -- NOT a learned stacker. Learned stacking was falsified in this
project (0.6912 vs 0.7079, p=0.0008); the failure mode there was a meta-model
fitting the OOF/meta distribution shift. Three global scalars cannot do that.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
OUT = ROOT / "5_outputs/ensemble_weighted.json"

DS = "Sheet2"
BASE = ["neural_2d", "traditional"]
CANDIDATES = ["xgb_atompair", "xgb_torsion", "svr_rbf_desc_morgan"]

PROTOCOLS = {"scaffold": (42, 1337, 2024), "butina": (42, 1337)}


def load(protocol, seeds):
    p = ROOT / f"5_outputs/arm_zoo_oof_{protocol}.npz"
    if not p.exists():
        raise SystemExit(f"missing {p}; run 4_scripts/arm_zoo.py first")
    z = np.load(p)
    store = {s: {k[len(f'{s}__'):]: z[k] for k in z.files if k.startswith(f"{s}__")}
             for s in seeds}
    folds = {s: json.loads(
        (ROOT / f"5_outputs/split_cache/{DS}_{protocol}_k5_seed{s}.json").read_text()
    )["folds"] for s in seeds}
    return store, folds


def _design(store, folds, seed, arms):
    A, b = [], []
    for f in folds[seed]:
        te = np.asarray(f["test"])
        A.append(np.stack([store[seed][a][te] for a in arms], 1))
        b.append(y[te])
    return np.vstack(A), np.concatenate(b)


def lopo(store, folds, seeds, arms):
    """Out-of-fold R2 per fold, with weights fitted only on other partitions."""
    out, weights = [], []
    for held in seeds:
        A, b = zip(*[_design(store, folds, t, arms) for t in seeds if t != held])
        w, _ = nnls(np.vstack(A), np.concatenate(b))
        weights.append((w / w.sum()).round(3).tolist())
        P = np.stack([store[held][a] for a in arms], 1)
        out += [r2_score(y[np.asarray(f["test"])], (P @ w)[np.asarray(f["test"])])
                for f in folds[held]]
    return np.asarray(out), weights


def equal_base(store, folds, seeds):
    out = []
    for s in seeds:
        P = (store[s]["neural_2d"] + store[s]["traditional"]) / 2
        out += [r2_score(y[np.asarray(f["test"])], P[np.asarray(f["test"])])
                for f in folds[s]]
    return np.asarray(out)


def main():
    global y
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)

    report = {"dataset": DS, "base": BASE, "protocols": {}}
    for protocol, seeds in PROTOCOLS.items():
        store, folds = load(protocol, seeds)
        base_eq = equal_base(store, folds, seeds)
        base_2, w2 = lopo(store, folds, seeds, BASE)
        print(f"\n===== {protocol}  leave-one-partition-out (n={len(base_eq)} folds) =====")
        print(f"  {'equal 2-arm (headline)':30s} {base_eq.mean():.4f} +- {base_eq.std(ddof=1):.4f}")
        t, p = ttest_rel(base_2, base_eq)
        print(f"  {'NNLS 2-arm (control)':30s} {base_2.mean():.4f} +- {base_2.std(ddof=1):.4f}"
              f"  delta={base_2.mean()-base_eq.mean():+.4f} p={p:.4f}"
              f"  wins={int((base_2 > base_eq).sum())}/{len(base_eq)}")

        entry = {"equal_2arm": {"mean": float(base_eq.mean()),
                                "std": float(base_eq.std(ddof=1))},
                 "nnls_2arm_control": {"mean": float(base_2.mean()),
                                       "delta": float(base_2.mean() - base_eq.mean()),
                                       "p": float(p), "weights": w2},
                 "candidates": {}}
        for cand in CANDIDATES:
            if cand not in store[seeds[0]]:
                continue
            c, w = lopo(store, folds, seeds, BASE + [cand])
            t, p = ttest_rel(c, base_eq)
            entry["candidates"][cand] = {
                "mean": float(c.mean()), "std": float(c.std(ddof=1)),
                "delta": float(c.mean() - base_eq.mean()), "p": float(p),
                "wins": int((c > base_eq).sum()), "n": int(len(c)), "weights": w}
            print(f"  {'NNLS +' + cand:30s} {c.mean():.4f} +- {c.std(ddof=1):.4f}"
                  f"  delta={c.mean()-base_eq.mean():+.4f} p={p:.4f}"
                  f"  wins={int((c > base_eq).sum())}/{len(c)}  w={w}")
        report["protocols"][protocol] = entry

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
