#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Progress + ETA for a queue of MolTrain runs, from the logs alone.

Written because the pKa queue is ~1 h per run and the only question during that
hour is "which fold is each run on, and when will it finish". Answering it by
grepping the log by hand is what this replaces.

Timing model, and why it is not just "elapsed / folds done":

* How many epochs a fold runs is set by EARLY STOPPING, not by ``--epochs``. On
  pKa a fold stops around epoch 10-15 of 40, so extrapolating to 40 overestimates
  a fold by roughly 3x. The estimator uses the epoch count actually observed.
* Fold 0 carries the one-time conformer pass, so its wall time is not
  representative and is excluded from the median when other folds exist.
* The current fold is estimated from its own per-epoch rate when enough epochs
  are logged (skipping its first epoch, the other outlier), and from the median
  fold otherwise.

This is a scheduling aid, not a number for a report -- no ETA appears in any
result file.

Usage:
    python 4_scripts/downstream/queue_status.py logs/downstream_pKa 'finetune_*_pKa_*.log'
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
FOLD_HEAD = re.compile(r"split fold (\d+): train=(\d+) test=(\d+)")
FOLD_DONE = re.compile(r"fold (\d+), result \{'r2': np\.float\d+\(([-0-9.eE+]+)\)")
EPOCH = re.compile(r"Epoch \[(\d+)/(\d+)\]")
DONE_LINE = re.compile(r"fold-mean R²=([-0-9.]+)±([-0-9.]+)")


def parse(path: Path) -> dict:
    """One pass over the log, attributing every epoch to the fold it belongs to.

    Epoch lines carry no fold id, and MolTrain prints ALL five ``split fold k:``
    announcements before any training starts. So neither "the last announcement
    seen" nor "the epoch's own index" identifies the fold: the first makes every
    run look like it is on fold 4, the second conflates fold 3 with epoch 3.

    Attribution is therefore positional over the training section: folds are
    trained in order, so the epochs before the first ``fold 0, result`` belong to
    fold 0, and each completion advances the cursor to the next fold.
    """
    text = path.read_text(errors="replace").replace("\r", "\n")
    folds: dict[int, dict] = {}
    cur: int | None = None
    started = False
    total_epochs = None
    for ln in text.split("\n"):
        m = TS.search(ln)
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None
        if mm := FOLD_HEAD.search(ln):
            k = int(mm.group(1))
            folds.setdefault(k, {"n_test": int(mm.group(3)), "epochs": [],
                                 "done": None, "r2": None, "t_start": None})
        elif (mm := EPOCH.search(ln)) and t:
            total_epochs = int(mm.group(2))
            if not started:
                # First epoch line: training has begun, so the active fold is the
                # first one (announcements are all printed up front).
                started, cur = True, 0
                folds.setdefault(0, {"n_test": None, "epochs": [], "done": None,
                                     "r2": None, "t_start": t})
            if cur is not None:
                if folds[cur]["t_start"] is None:
                    folds[cur]["t_start"] = t
                folds[cur]["epochs"].append((int(mm.group(1)), t))
        elif (mm := FOLD_DONE.search(ln)) and t:
            k = int(mm.group(1))
            folds.setdefault(k, {"n_test": None, "epochs": [], "done": None,
                                 "r2": None, "t_start": None})
            folds[k].update(done=t, r2=float(mm.group(2)))
            cur = k + 1  # folds train in order; the next unfinished one is active
    summary = None
    if (mm := DONE_LINE.search(text)):
        summary = (float(mm.group(1)), float(mm.group(2)))
    return {"path": path, "folds": folds, "total_epochs": total_epochs,
            "summary": summary, "mtime": path.stat().st_mtime}


def current_fold(folds: dict) -> int | None:
    """The fold in flight: the first one that has been announced and not finished.

    Not "the max unfinished fold". MolTrain prints all five ``split fold k:``
    lines BEFORE training starts, so every fold is announced from the beginning
    and "max unfinished" is always the last one -- which reports a run at fold 0
    as being at fold 4 and makes every ETA meaningless. Folds are started in
    order, so the first unfinished one is the active one.
    """
    for k in sorted(folds):
        if folds[k]["done"] is None:
            return k
    return None


def describe(s: dict, kfold: int = 5, epochs_total: int = 40) -> str:
    name = s["path"].stem
    folds = s["folds"]
    n_done = sum(1 for f in folds.values() if f["done"] is not None)
    total_epochs = s["total_epochs"] or epochs_total

    if s["summary"] is not None and n_done >= kfold:
        r2 = [folds[k]["r2"] for k in sorted(folds) if folds[k]["r2"] is not None]
        return (f"{name:<40} DONE  fold-mean R2={s['summary'][0]:.4f} "
                f"per-fold={[round(x, 4) for x in r2]}")

    cur = current_fold(folds)
    if cur is None:
        return f"{name:<40} starting (no fold announced yet)"

    # Median wall time of COMPLETED folds, excluding fold 0 (one-time conformer
    # pass) whenever a later fold has also finished.
    done_ks = sorted(k for k, f in folds.items() if f["done"] is not None)
    walls = [(k, (folds[k]["done"] - folds[k]["t_start"]).total_seconds())
             for k in done_ks
             if folds[k]["t_start"] is not None]
    walls = [(k, w) for k, w in walls if w > 0]
    pool = [w for k, w in walls if k != 0] or [w for _k, w in walls]
    med_wall = sorted(pool)[len(pool) // 2] if pool else None

    # Epochs a fold actually runs: the max reached by completed folds.
    lens = [max(e for e, _t in folds[k]["epochs"]) for k in done_ks
            if folds[k]["epochs"]]
    exp_epochs = sorted(lens)[len(lens) // 2] if lens else total_epochs

    eps = folds[cur]["epochs"]
    last = eps[-1] if eps else None
    # Skip the current fold's first epoch: it includes the feature build.
    per_ep = None
    if len(eps) >= 3:
        per_ep = ((eps[-1][1] - eps[1][1]).total_seconds()
                  / max(eps[-1][0] - eps[1][0], 1))
    if per_ep is not None and last is not None:
        # At least one more epoch is coming -- early stopping has not fired yet,
        # so a fold that has already passed the median epoch count is not at zero
        # remaining. Flooring at 1 avoids printing an ETA of "0.0 min" for a fold
        # that then runs another ten minutes.
        eta_here = max(exp_epochs - last[0], 1) * per_ep
    else:
        eta_here = med_wall
    eta = eta_here + med_wall * (kfold - cur - 1) if (
        eta_here is not None and med_wall is not None) else None

    idle = time.time() - s["mtime"]
    flag = "  <-- STALLED?" if idle > 900 else ""
    return (f"{name:<40} fold {cur}/{kfold}  epoch "
            f"{(last[0] if last else 0):>2}/{exp_epochs}  "
            f"fold~{med_wall/60:.1f}min  " if med_wall else
            f"{name:<40} fold {cur}/{kfold}  epoch "
            f"{(last[0] if last else 0):>2}/{exp_epochs}  fold~?  ") + (
        f"ETA {eta/60:5.1f} min  " if eta is not None else "ETA   ?     ") + (
        f"last-log {idle/60:5.1f} min ago{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", type=Path)
    ap.add_argument("pattern", nargs="?", default="*.log")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    paths = sorted(Path(a.logdir).glob(a.pattern))
    if not paths:
        raise SystemExit(f"no logs match {a.logdir}/{a.pattern}")
    print(f"=== {len(paths)} logs in {a.logdir}  ({time.strftime('%H:%M:%S')}) ===")
    for p in paths:
        try:
            print(describe(parse(p), a.kfold, a.epochs))
        except Exception as exc:  # a log mid-write must not kill the report
            print(f"{p.stem:<40} (unreadable: {type(exc).__name__}: {exc})")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print("--- gpu ---")
    print(out.stdout.strip() or "(nvidia-smi unavailable)")


if __name__ == "__main__":
    main()
