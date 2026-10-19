#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Before spending anything on a (mu, log sigma) head: does the EARLY STOPPING survive it?

The idea under test
-------------------
unimol_tools already ships a task named `multilabel_regression` that accepts an
(n, k) target and applies one TargetScaler per column (datascaler.py:60-72, 164-176).
If that path works unmodified, the regression head can be made to emit
(mu, log sigma) by changing the TARGET SHAPE only -- no new task name, no edits to
the three registries in nnmodel.py, no metrics.py registration. That would make the
distillation route materially cheaper than the handoff assumed.

The specific risk
-----------------
MolTrain chooses its early-stopping metric as
`list(metric_score.values())[0]` (tasks/trainer.py:500), i.e. the FIRST metric of the
task's default list, and evaluates it through `cal_nan_metric`, which loops over the
k columns and returns their MEAN (utils/metrics.py:43-49). For k=2 that means the
number driving early stopping is "mse averaged over (mu, log sigma)":

  * the sigma column lives on a completely different scale from mu (log sigma is
    order 0, mu is order -3 in z-space), so the average is dominated by whichever
    column is larger -- not by what we care about;
  * worse, it can be gamed: the model can lower the averaged mse by shrinking the
    sigma column toward its mean, which is exactly the opposite of learning
    per-molecule uncertainty.

So the question is not "does multilabel_regression run" -- it will -- but "is the
checkpoint it selects the one that predicts mu well".

What this script does
---------------------
Runs a REAL MolTrain fit for one fold of scaffold/seed42, twice:
  A) task='regression', 1-column target          -- the reference (this IS the
     production configuration; its fold-0 R2 must match the historical run)
  B) task='multilabel_regression', 2-column target (y, teacher_log_sigma)
and reports, for BOTH, the per-epoch history of:
     - the metric MolTrain actually early-stops on
     - the R2 of the mu column alone
so we can see whether the selected checkpoint is chosen for the right reason.
It also reports whether any of the two silent traps from the handoff fire.

Folds are the cached ones (--kfold 1 with an injected single fold) so the run
matches the production split exactly and finishes in ~10 min per config.

Usage:
    python 4_scripts/earlystop_probe.py --config both
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
FEAT = ROOT / "5_outputs/feat_cache"

DS = "Sheet2"
SPLIT, SEED, FOLD = "scaffold", 42, 0
CKPT = ROOT / ("2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/"
               "checkpoint_1_30000.pt")
OUT = ROOT / "5_outputs/earlystop_probe.json"


# ------------------------------------------------------- teacher sigma (column 2)
def teacher_log_sigma(smis: list[str]) -> np.ndarray:
    """The validated epistemic sigma, regressed onto fingerprints.

    Uses the same estimator validated in `sigma_predictability.py`: log(sigma) is
    learnable from structure (RF, fold-mean R2 +0.309). Fitting it here on the
    FOLD'S TRAIN SPLIT only -- the probe must not hand the student a sigma derived
    from its own test molecules, or config B would be scored under a leak the
    production setting would not have.
    """
    X = np.hstack([np.load(FEAT / f"{DS}_{k}.npy")
                   for k in ("rdkit_desc", "morgan_r2_2048", "maccs",
                             "atompair", "torsion")])
    z = np.load(ROOT / "5_outputs/ensemble_oof.npz")
    panel = z["scaffold__panel"]                 # (n_arms, n_partitions, n)
    sigma = panel.mean(1).std(0, ddof=1)
    return X, np.log(sigma)


def fold_indices():
    f = json.loads((ROOT / f"5_outputs/split_cache/{DS}_{SPLIT}_k5_seed{SEED}"
                    ".json").read_text())["folds"][FOLD]
    return np.asarray(f["train"], int), np.asarray(f["test"], int)


def run_config(tag: str, target: np.ndarray, task: str) -> dict:
    """One MolTrain fit on the fold, returning the full per-epoch history."""
    import finetune_moltrain as fm
    fm.USE_STOCK_84M = False
    fm.CHECKPOINT = CKPT
    fm._install_lipidfm_loader()
    from unimol_tools import MolTrain

    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    tr, te = fold_indices()

    # One fold, injected through the project's own fold injector so the split is
    # byte-identical to the cached one.
    fm._install_fold_injector([(tr, te)])

    # The vendored unimol_tools 0.1.6 has NO DataFrame branch in read_data
    # (CLAUDE.md 坑 #1). finetune_moltrain passes a dict; do the same. Every entry
    # must be 1-D -- read_data does `pd.DataFrame(data)`, which rejects per-column
    # arrays of shape (n, 1) with "Per-column arrays must each be 1-dimensional".
    # So the multi-column config passes one flat key per column, and only the single
    # column config uses the name 'TARGET'.
    if target.shape[1] == 1:
        payload = {"SMILES": smis, "TARGET": target[:, 0]}
        cols = "TARGET"
    else:
        payload = {"SMILES": smis}
        for j in range(target.shape[1]):
            payload[f"T{j}"] = target[:, j]
        cols = ",".join(f"T{j}" for j in range(target.shape[1]))

    out_dir = ROOT / f"5_outputs/earlystop_probe/{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clf = MolTrain(
        task=task, data_type="molecule", epochs=8, learning_rate=5e-5, batch_size=16,
        early_stopping=99,            # disable so the FULL history is recorded
        metrics="r2", split="random", kfold=1, save_path=str(out_dir),
        model_name="unimolv2", model_size="84m", smiles_col="SMILES",
        target_cols=cols, target_normalize="standard", use_cuda=True, use_amp=True,
        conf_cache_level=1, max_atoms=256,
    )
    clf.config["seed"] = SEED
    clf.config["split_seed"] = SEED
    clf.fit(payload)
    dt = time.time() - t0

    # SCALE CONTRACT (CLAUDE.md 坑 #4, and the comment block in finetune_moltrain.py
    # that documents this exact trap): `clf.cv_pred` has ALREADY been un-standardised
    # inside MolTrain.fit() (train.py:191-203) and is on the MODELLING scale.
    # Applying inverse_transform a second time re-inflates it by (mu + sd*x): ranks
    # survive but the scale does not, which is how this produced R2 = -25.59 on the
    # first run. Only `clf.data['target']` still needs the inverse transform.
    pred_m = np.asarray(clf.cv_pred, dtype=float)
    if pred_m.ndim == 1:
        pred_m = pred_m.reshape(-1, 1)
    mu_col = pred_m[:, 0]
    # The scaler is a LIST of per-column scalers under multilabel_regression and a
    # single scaler under regression (datascaler.py:60-72). The list branch asserts
    # len(scaler) == target.shape[1], so each column must be passed through its own
    # scaler -- passing the array whole raises AssertionError.
    sc = clf.data["target_scaler"]
    tgt_z = np.asarray(clf.data["target"])
    if tgt_z.ndim == 1:
        tgt_z = tgt_z.reshape(-1, 1)
    if isinstance(sc.scaler, list):
        target_m = np.column_stack([
            sc.inverse_transform(tgt_z[:, j:j + 1]).ravel()
            for j in range(tgt_z.shape[1])])
    else:
        target_m = sc.inverse_transform(tgt_z)
    print(f"    [scale contract] cv_pred range "
          f"[{pred_m.min():.2f}, {pred_m.max():.2f}]  (log2 RLU scale, not z)")
    print(f"    [scaler] {'list of ' + str(len(sc.scaler)) if isinstance(sc.scaler, list) else 'single'}")

    # per-epoch trace from the saved training log if present, else from metric.result
    hist = getattr(clf, "train_history", None)
    return {
        "task": task, "n_target_cols": int(target.shape[1]), "seconds": round(dt, 1),
        "fold0_mu_r2": float(r2_score(y[te], mu_col[te])),
        "fold0_all_cols_r2": [float(r2_score(target_m[te, j], pred_m[te, j]))
                              for j in range(target.shape[1])],
        "metric_result": {k: float(v) for k, v in
                          clf.cv["metric"].items()} if hasattr(clf, "cv") else None,
        "history_available": hist is not None,
        "n_epochs_ran": int(len(hist)) if hist else None,
        "epoch_history": hist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    tr, te = fold_indices()

    X, tlog = teacher_log_sigma(smis)
    # teacher sigma fitted on THIS FOLD'S TRAIN ONLY (no leakage into the test fold)
    t_hat = np.full(len(y), np.nan)
    t_hat[tr] = RandomForestRegressor(n_estimators=400, max_features=0.3, n_jobs=12,
                                      random_state=0).fit(X[tr], tlog[tr]).predict(X[tr])
    print(f"teacher log(sigma) on train fold: mean={t_hat[tr].mean():.3f} "
          f"std={t_hat[tr].std(ddof=1):.3f}")

    report = {"dataset": DS, "split": SPLIT, "seed": SEED, "fold": FOLD,
              "n_train": int(len(tr)), "n_test": int(len(te)),
              "tested": "early-stopping metric under a 2-column target", "runs": {}}

    if args.config in ("A", "both"):
        print("\n=== A: task=regression, 1 column (reference) ===", flush=True)
        report["runs"]["A_regression_1col"] = run_config("A_regression_1col",
                                                         y.reshape(-1, 1), "regression")
        r = report["runs"]["A_regression_1col"]
        print(f"  fold0 mu R2 = {r['fold0_mu_r2']:.4f}   ({r['seconds']}s)")

    if args.config in ("B", "both"):
        print("\n=== B: task=multilabel_regression, 2 columns (y, log sigma) ===",
              flush=True)
        tgt = np.column_stack([y, t_hat])
        report["runs"]["B_multilabel_2col"] = run_config("B_multilabel_2col", tgt,
                                                         "multilabel_regression")
        r = report["runs"]["B_multilabel_2col"]
        print(f"  fold0 mu R2 = {r['fold0_mu_r2']:.4f}   "
              f"sigma-col R2 = {r['fold0_all_cols_r2'][1]:.4f}   ({r['seconds']}s)")

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
