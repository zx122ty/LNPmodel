#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure what one fold of each classical arm costs, on a named dataset.

Why a script and not an extrapolation
-------------------------------------
The plan (``prompt/26-09-16/01-*`` §7) budgets evaluation 1 as "~14 GPU·h + CPU"
and leaves the CPU side as a dash. That was defensible while the baselines had
only ever run on Sheet2. It stopped being defensible on 2026-09-18, when the pKa
run produced the first per-arm timings for the *classical* arms:

    ecfp4_rf    2.2 s/seed
    ecfp4_xgb   749-1083 s/seed      <- 400x the RF
    mlp_morgan  198-334 s/seed

Those are on 574 molecules. Evaluation 1's Sheet2 is 1920 -- 3.3x the data, and
gradient boosting is worse than linear in rows. A 3.3x multiplier on 80.7 min is
4.4 h *per seed*, i.e. 22 h for one arm's 5-seed grid, and evaluation 1 wants
three arms across two datasets. That is a planning number, not a footnote.

So: measure one fold per arm on the dataset in question. One fold, not a full
grid, because the point is to size the queue before committing to it.

Why this reports CPU-seconds, not just wall clock
-------------------------------------------------
The first version of this script reported wall clock only, and on 2026-09-18 it
produced a number that was wrong by 24x: it timed ``ecfp4_xgb`` on pKa/butina at
2.3-4.5 s/fold while the stored result cell for the *same folds* recorded
749.0 s. The gap is not a mystery -- it is contention. ``xgboost`` and
``sklearn`` both grab every core by default (``n_jobs=-1``), so wall clock here
is ``cpu_seconds / cores_actually_granted``. The probe ran on an idle box and
got ~24 cores' worth; the recorded run shared the machine with a 3-GPU MolTrain
queue and got ~1 core's worth. Same work, same folds (the probe's per-fold R2
matched ``r2_per_fold`` exactly, which is what proved it was the same work).

Wall clock is the wrong unit for sizing a queue that will itself be contended,
so the primary number is now ``cpu_seconds_per_fold`` -- the process CPU time,
which is invariant to what else is on the box. Wall clock is still reported
next to it, because that is what the queue will actually experience.

``--n-folds 5`` is also now the default. One fold is enough to size an idle box
but not enough to see the contention spread: the same arm on the same data
ranged 1.9-12.8 s across folds in the 2026-09-18 measurement, a 6.7x spread that
a single fold samples arbitrarily.

Usage:
    python 4_scripts/downstream/cost_probe.py --dataset Sheet2 --split butina
    python 4_scripts/downstream/cost_probe.py --dataset Sheet3 --split butina --arms ecfp4_rf mlp_morgan
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines  # noqa: E402
import ds_registry  # noqa: E402
import split_files  # noqa: E402

OUT = ROOT / "5_outputs/downstream_results/cost_probe.json"


def time_arm(model: str, X: np.ndarray, y: np.ndarray, folds, target_name: str,
             n_folds: int) -> dict:
    """Fit/predict the first ``n_folds`` folds and report seconds per fold.

    Only the first fold is used by default. Folds are the same size to within a
    molecule or two on every split here, so one is enough to size a queue --
    and running five would take as long as the thing being planned.
    """
    from sklearn.metrics import r2_score

    factory = baselines.REGISTRY[model]
    baselines.bind_target(y)
    times, cpus, r2s = [], [], []
    for k, (tr, te) in enumerate(folds[:n_folds]):
        tr = np.asarray(tr, int)
        te = np.asarray(te, int)
        # The factories return a `fit_predict(tr, te)` closure rather than an
        # sklearn estimator -- that is the interface `run_matrix` consumes, and
        # timing anything else would measure a code path the queue does not run.
        fit_predict = factory(X)
        c0, t0 = time.process_time(), time.time()
        pred = fit_predict(tr, te)
        wall = time.time() - t0
        cpu = time.process_time() - c0
        times.append(wall)
        cpus.append(cpu)
        r2s.append(float(r2_score(y[te], pred[:, 0] if pred.ndim > 1 else pred)))
    wall_mean = float(np.mean(times))
    cpu_mean = float(np.mean(cpus))
    return {
        "model": model, "target": target_name, "folds_timed": len(times),
        # cpu_seconds_per_fold is the portable number: it does not change with
        # what else is running. seconds_per_fold is what the queue will feel.
        "cpu_seconds_per_fold": round(cpu_mean, 1),
        "seconds_per_fold": round(wall_mean, 1),
        # ~1.0 means the arm got a single core; a large value means it really
        # did use that many, and wall clock on an idle box will scale with it.
        "effective_cores": round(cpu_mean / wall_mean, 1) if wall_mean else None,
        "seconds_each": [round(t, 1) for t in times],
        "cpu_seconds_each": [round(c, 1) for c in cpus],
        "r2_per_fold": [round(r, 4) for r in r2s],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(ds_registry.LOADERS))
    ap.add_argument("--split", default="butina")
    ap.add_argument("--arms", nargs="+", default=["ecfp4_rf", "mlp_morgan", "ecfp4_xgb"],
                    choices=sorted(baselines.REGISTRY))
    ap.add_argument("--seed", type=int, default=None,
                    help="partition seed (default: the first frozen seed for this dataset/split)")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5,
                    help="how many folds to time per arm (default 5: one fold "
                         "cannot see the per-fold spread, which reached 6.7x)")
    ap.add_argument("--json-out", type=Path, default=OUT)
    args = ap.parse_args()

    df, target_cols = ds_registry.load(args.dataset, verbose=True)
    smiles = df["SMILES"].astype(str).tolist()
    y = df[target_cols[0]].to_numpy(float)
    seeds = split_files.seeds_for(args.dataset, args.split)
    seed = args.seed if args.seed is not None else seeds[0]
    folds = split_files.load_folds(args.dataset, args.split, seed, args.kfold, smiles=smiles)

    t0 = time.time()
    X = baselines.morgan_fingerprints(smiles)
    fp_secs = time.time() - t0
    print(f"  n={len(smiles)}  fingerprints {X.shape} in {fp_secs:.1f}s  "
          f"split={args.split} seed={seed}  timing {args.n_folds} fold(s)/arm\n")

    rec = {"dataset": args.dataset, "split": args.split, "seed": seed,
           "kfold": args.kfold, "n": len(smiles), "n_targets": len(target_cols),
           "target": target_cols[0], "fingerprint_seconds": round(fp_secs, 1),
           # Recorded because wall clock is only interpretable against it: the
           # 2026-09-18 discrepancy was entirely explained by this number.
           "loadavg_at_probe": [round(x, 2) for x in os.getloadavg()],
           "nproc": os.cpu_count(),
           "arms": {}}

    for model in args.arms:
        if model == "mlp_morgan" and len(target_cols) != 1:
            print(f"  {model}: skipped (single-target arm, dataset has "
                  f"{len(target_cols)} targets)")
            continue
        r = time_arm(model, X, y, folds, target_cols[0], args.n_folds)
        rec["arms"][model] = r
        print(f"  {model:<12} {r['cpu_seconds_per_fold']:>8.1f} cpu-s/fold   "
              f"{r['seconds_per_fold']:>7.1f} s wall   "
              f"({r['effective_cores']} cores)   R2={r['r2_per_fold'][0]:+.4f}")

    n_seeds = len(seeds)
    n_folds = n_seeds * args.kfold
    print(f"\n=== projected full grid for {args.dataset}/{args.split} "
          f"({n_seeds} seeds x {args.kfold} folds = {n_folds} folds) ===")
    print(f"  {'arm':<12} {'cpu-hours':>10} {'wall-hours on N cores':>22}")
    tot_cpu = 0.0
    for model, r in rec["arms"].items():
        cpu_h = r["cpu_seconds_per_fold"] * n_folds / 3600
        tot_cpu += cpu_h
        wall = (cpu_h / r["effective_cores"]) if r["effective_cores"] else float("nan")
        print(f"  {model:<12} {cpu_h:>10.2f} {wall:>22.2f}   (at {r['effective_cores']} cores)")
    print(f"  {'TOTAL':<12} {tot_cpu:>10.2f}")
    rec["projected_cpu_hours"] = {m: round(r["cpu_seconds_per_fold"]*n_folds/3600, 2)
                                  for m, r in rec["arms"].items()}
    rec["projected_cpu_hours_total"] = round(tot_cpu, 2)
    rec["n_seeds"] = n_seeds
    rec["n_folds_total"] = n_folds

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.json_out.exists():
        existing = json.loads(args.json_out.read_text(encoding="utf-8"))
    existing[f"{args.dataset}|{args.split}|seed{seed}"] = rec
    args.json_out.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
