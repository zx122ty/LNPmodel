#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrator for the downstream evaluation grid: model x split x seed.

One rule drives this module: **the same fold file must reach every model.**
The project has already paid for ignoring it once -- the stock baseline used
MolTrain's own splitter instead of the cached folds, and stock 'scaffold'
(GroupKFold with one giant group for acyclic lipids) agreed with the real
assignment only ~5% of the time. Comparing those two numbers is meaningless.

So this module never lets a model compute its own split. It reads the frozen
JSON written by ``split_files.py`` and hands the model an explicit train/test
index pair per fold.

The second rule is that a run is identified by every variable that can change
its result (the project's 坑 #6). ``run_id`` is built from all of them here, in
one place, because an aggregate key that omits one has already caused a new run
to silently overwrite an old one's evidence.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import split_files  # noqa: E402
from metrics import paired_compare, score_folds  # noqa: E402

RESULTS_DIR = ROOT / "5_outputs/downstream_results"
PRED_DIR = ROOT / "5_outputs/downstream_predictions"


def run_id(dataset: str, model: str, split: str, seed: int, kfold: int,
           **extra) -> str:
    """Stable identifier covering every variable that can change the result.

    ``extra`` absorbs model-specific hyperparameters. Anything that changes what
    is trained belongs in the key; anything that only changes logging does not.
    """
    parts = [dataset, model, split, f"seed{seed}", f"k{kfold}"]
    for k in sorted(extra):
        v = extra[k]
        if v is not None:
            parts.append(f"{k}{v}")
    return "|".join(parts)


def run_sklearn_model(fit_predict, df: pd.DataFrame, target_cols: list[str],
                      folds: list, dataset: str, model: str, split: str,
                      seed: int, **extra) -> dict:
    """Evaluate one (model, dataset, split, seed) cell and persist its predictions.

    ``fit_predict(train_idx, test_idx, X_target) -> np.ndarray`` returning
    predictions for the TEST rows, shape ``(len(test), n_targets)``.

    Predictions are written per cell before scoring, so a scoring bug never
    costs a refit -- and so the plan's "publish per-sample predictions" item is
    satisfied by construction rather than by a later export step.
    """
    rid = run_id(dataset, model, split, seed, len(folds), **extra)
    y = df[target_cols].to_numpy(float)
    n, n_t = len(df), len(target_cols)
    oof = np.full((n, n_t), np.nan)

    t0 = time.time()
    for fi, (tr, te) in enumerate(folds):
        tr, te = np.asarray(tr, int), np.asarray(te, int)
        pred = np.asarray(fit_predict(tr, te), float)
        if pred.shape != (len(te), n_t):
            raise SystemExit(
                f"{rid} fold {fi}: fit_predict returned {pred.shape}, expected "
                f"{(len(te), n_t)}. A transposed or truncated prediction would "
                f"still score -- refusing instead.")
        oof[te] = pred

    uncovered = int(np.isnan(oof[:, 0]).sum())
    if uncovered:
        # Every molecule must be predicted exactly once by the fold that held
        # it out; anything else means the fold list does not cover the data.
        raise SystemExit(f"{rid}: {uncovered}/{n} molecules never landed in a test "
                         f"fold -- folds do not cover the dataset")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_DIR / f"{rid}.npz", oof=oof, y=y,
                        targets=np.array(target_cols, dtype=object))

    per_target = {}
    for j, tname in enumerate(target_cols):
        per_target[tname] = score_folds(y[:, j], oof[:, j], folds)
    summary = {
        "run_id": rid, "dataset": dataset, "model": model, "split": split,
        "seed": seed, "kfold": len(folds), "n": n, "targets": target_cols,
        "elapsed_sec": round(time.time() - t0, 1), "per_target": per_target,
        **extra,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{rid}.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def compare_runs(ours_path: Path, baseline_path: Path, metric: str = "r2",
                 target: str | None = None) -> dict:
    """Paired Wilcoxon between two runs on the SAME folds.

    Refuses to compare runs from different splits or fold counts: a paired test
    across mismatched folds is not a conservative version of the test, it is a
    different and meaningless one.
    """
    a = json.loads(Path(ours_path).read_text(encoding="utf-8"))
    b = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    for field in ("dataset", "split", "kfold"):
        if a[field] != b[field]:
            raise SystemExit(
                f"cannot pair: {field} differs ({a[field]!r} vs {b[field]!r}). "
                f"Paired tests require identical folds.")
    tgt = target or a["targets"][0]
    ours = a["per_target"][tgt][f"{metric}_per_fold"]
    base = b["per_target"][tgt][f"{metric}_per_fold"]
    return {"target": tgt, "metric": metric, "ours": a["run_id"],
            "baseline": b["run_id"],
            **paired_compare(ours, base, name=b["model"])}


def main() -> None:
    print("downstream/run_matrix.py is a library; import it from a driver script.")
    print(f"  frozen splits : {split_files.SPLIT_DIR}")
    print(f"  results       : {RESULTS_DIR}")
    print(f"  predictions   : {PRED_DIR}")


if __name__ == "__main__":
    main()
