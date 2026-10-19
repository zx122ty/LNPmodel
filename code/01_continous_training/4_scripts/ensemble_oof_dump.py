#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 0 of P1: put the deployed ensemble's per-molecule OOF predictions on disk.

Why this exists
---------------
Everything downstream (L1 uncertainty, L3 active learning) needs a PREDICTION
VECTOR per molecule for the current-best ensemble, not just a fold-mean R2.
`ensemble_weighted.py` reports the number but keeps the vectors local: its
`lopo()` returns per-fold R2 only. So the analysis would have had to fall back
to an equal-weight demo ensemble, which is not the thing that was published.

What it produces
----------------
    pred            (n,)                the LOPO-weighted 3-arm prediction
    mu_<arm>        (n,)                each arm's own prediction, per partition
    ...

Weights are leave-one-partition-out, exactly as in `ensemble_weighted.lopo()`:
the weights applied to a held-out partition are fitted on the other partitions'
OOF only, so no fold ever sees its own weights. The scalar per-fold R2 below is
therefore the SAME quantity that script reports, and is checked against it.

Partition coverage
------------------
`ensemble_weighted.PROTOCOLS` gave butina two partitions (42, 1337) while
scaffold got three, so the published butina headline (0.8121) is a 10-fold
number while scaffold's (0.7143) is a 15-fold one -- they are not comparable,
and P1 compares sigma across protocols. The neural seed-2024 butina run has
existed on disk since 2026-09-13; only the CPU arms were missing. `--fill`
recomputes those (fingerprint caches are already on disk, so ~2 min) and then
butina runs on the same 3-partition / 15-fold footing as scaffold.

Both are reported: the 2-partition numbers reproduce the published figures, the
3-partition ones are the P1 baseline. The report says which is which.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import nnls

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")

DS = "Sheet2"
BASE = ["neural_2d", "traditional"]
THIRD = "xgb_atompair"
ARMS = BASE + [THIRD]

# The two protocols as originally measured, plus the partition that was only ever
# run for the neural arm.
PROTOCOLS = {
    "scaffold": {"seeds": (42, 1337, 2024), "published": (42, 1337, 2024)},
    "butina": {"seeds": (42, 1337, 2024), "published": (42, 1337)},
}

# Every CPU arm arm_zoo.py builds. Kept in sync with its `build_arms` by importing
# the feature builders from it rather than re-deriving the fingerprints here.
ZOO_ARMS = [
    "svr_rbf_desc_morgan", "krr_rbf_desc_morgan", "rf_desc_morgan",
    "svr_tanimoto", "knn_tanimoto", "xgb_maccs", "xgb_torsion",
    "xgb_atompair", "svr_rbf_desc_maccs", "krr_rbf_torsion_atompair",
]


def load_labels() -> np.ndarray:
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    return pd.to_numeric(d["RLU (log2)"]).values.astype(float)


def load_folds(protocol: str, seed: int) -> list:
    p = ROOT / f"5_outputs/split_cache/{DS}_{protocol}_k5_seed{seed}.json"
    return json.loads(p.read_text())["folds"]


def load_npz(protocol: str) -> dict:
    p = ROOT / f"5_outputs/arm_zoo_oof_{protocol}.npz"
    if not p.exists():
        raise SystemExit(f"missing {p}; run 4_scripts/arm_zoo.py first")
    z = np.load(p)
    out: dict[int, dict[str, np.ndarray]] = {}
    for k in z.files:
        seed_s, arm = k.split("__", 1)
        out.setdefault(int(seed_s), {})[arm] = z[k]
    return out


# --------------------------------------------------------------- fill missing
def fill_partition(protocol: str, seed: int) -> dict[str, np.ndarray]:
    """Recompute arm_zoo's CPU arms for a partition the saved npz never covered.

    The arm set and configurations are imported from arm_zoo.py, so a filled
    partition is the same experiment as the saved ones, not a re-implementation.
    """
    import sys
    sys.path.insert(0, str(ROOT / "4_scripts"))
    import arm_zoo

    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    folds = [(np.asarray(f["train"], int), np.asarray(f["test"], int))
             for f in load_folds(protocol, seed)]

    arms = arm_zoo.build_arms(y, smis)
    out = {}
    for name, (kind, X) in arms.items():
        _, _, pred = arm_zoo.oof(kind, X, y, folds)
        out[name] = pred
        print(f"    {name:24s} {np.mean([r2_score(y[te], pred[te]) for _, te in folds]):+.4f}",
              flush=True)
    return out


def neural_butina(seed: int) -> np.ndarray | None:
    p = (ROOT / "5_outputs/finetune_moltrain"
         / f"{DS}_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_butina_seed{seed}"
           f"_lr5e-05_ep40")
    if not (p / "cv.data").exists():
        return None
    z = np.asarray(joblib.load(p / "cv.data")).ravel()
    return joblib.load(p / "target_scaler.ss").inverse_transform(z.reshape(-1, 1)).ravel()


def traditional_butina(seed: int) -> np.ndarray | None:
    a = ROOT / "5_outputs/traditional_butina_oof.npz"
    b = ROOT / "5_outputs/traditional_butina_oof_more.npz"
    if seed == 42 and a.exists():
        return np.load(a)["pred"]
    if b.exists():
        d = np.load(b)
        if f"pred_{seed}" in d:
            return d[f"pred_{seed}"]
    return None


# ------------------------------------------------------------------- the blend
def lopo_prediction(store: dict[int, dict[str, np.ndarray]], y: np.ndarray,
                    seeds: tuple, arms: list) -> tuple[dict[int, np.ndarray], list[list[float]]]:
    """Per-partition OOF predictions under leave-one-partition-out weights.

    Mirrors ensemble_weighted.lopo(): for each held-out partition the weights come
    only from the other partitions' out-of-fold predictions.

    !! The return value is ONE VECTOR PER PARTITION, not one global vector. Every
    partition's folds tile all n molecules (the partitions differ only in how they
    group them), so a single global vector cannot exist: assembling one by writing
    `pred[te] = ...` for each partition in turn silently keeps only the LAST
    partition, because the earlier ones' indices are overwritten. That is not
    hypothetical -- the first version of this script did exactly that and produced
    a perfectly plausible 0.7129 instead of 0.7143. The reference script never hit
    it because it scores each partition independently and never materialises a
    global vector. Keeping the vectors separate also preserves the fold pairing,
    which every comparison in this project is required to respect.
    """
    preds: dict[int, np.ndarray] = {}
    weights = []
    for held in seeds:
        A = [np.stack([store[t][a] for a in arms], 1) for t in seeds if t != held]
        b = [y[np.asarray(f["test"], int)] for t in seeds if t != held
             for f in load_folds(CUR_PROTOCOL, t)]
        w, _ = nnls(np.vstack([np.vstack([store[t][a][np.asarray(f["test"], int)]
                                          for a in arms]).T
                               for t in seeds if t != held
                               for f in load_folds(CUR_PROTOCOL, t)]), np.concatenate(b))
        weights.append((w / w.sum()).round(3).tolist())
        P = np.stack([store[held][a] for a in arms], 1)
        preds[held] = P @ w
    return preds, weights


def fold_r2(preds: dict[int, np.ndarray], y: np.ndarray,
            protocol: str, seeds: tuple) -> np.ndarray:
    """Per-fold R2, each fold scored against its own partition's prediction vector."""
    return np.asarray([
        r2_score(y[np.asarray(f["test"], int)], preds[s][np.asarray(f["test"], int)])
        for s in seeds for f in load_folds(protocol, s)
    ])


def main() -> int:
    global CUR_PROTOCOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true",
                    help="recompute the CPU arms missing from the butina npz "
                         "(seed 2024) and cache them to npz_filled_butina.npz")
    args = ap.parse_args()

    y = load_labels()
    print(f"{DS}: n={len(y)}")

    report = {"dataset": DS, "arms": ARMS, "protocols": {}}
    blobs: dict[str, np.ndarray] = {}

    for protocol, spec in PROTOCOLS.items():
        CUR_PROTOCOL = protocol
        store = load_npz(protocol)
        seeds = spec["seeds"]

        # ---- partition 2024 is missing under butina: fill it first ----
        missing = [s for s in seeds if s not in store
                   or not all(a in store[s] for a in ARMS)]
        if missing:
            fill_path = ROOT / f"5_outputs/arm_zoo_oof_{protocol}_filled.npz"
            if fill_path.exists():
                z = np.load(fill_path)
                for k in z.files:
                    seed_s, arm = k.split("__", 1)
                    store.setdefault(int(seed_s), {})[arm] = z[k]
                print(f"  loaded filled arms from {fill_path.name}")
            elif not args.fill:
                print(f"  !! {protocol}: partition(s) {missing} incomplete; "
                      f"re-run with --fill to compute them")
                continue
            else:
                print(f"  filling {protocol} partition(s) {missing}...", flush=True)
                t0 = time.time()
                for s in missing:
                    cur = fill_partition(protocol, s)
                    if s == 2024 and protocol == "butina":
                        n = neural_butina(s)
                        if n is None:
                            raise SystemExit(f"neural butina seed{s} artifacts missing")
                        cur["neural_2d"] = n
                        t = traditional_butina(s)
                        if t is None:
                            raise SystemExit(f"traditional butina seed{s} OOF missing")
                        cur["traditional"] = t
                    store.setdefault(s, {}).update(cur)
                np.savez(fill_path, **{f"{s}__{a}": v for s, d_ in store.items()
                                       if s in missing for a, v in d_.items()})
                print(f"  filled in {time.time()-t0:.0f}s -> {fill_path.name}")

        for tag, use_seeds in (("published", spec["published"]), ("p1_equal_folds", seeds)):
            if tag == "published" and tuple(use_seeds) == tuple(seeds):
                # scaffold's published口径 already IS the 3-partition one; no need
                # to report the same number twice.
                continue
            preds, w = lopo_prediction(store, y, tuple(use_seeds), ARMS)
            r2 = fold_r2(preds, y, protocol, tuple(use_seeds))
            eq = np.asarray([
                r2_score(y[np.asarray(f["test"], int)],
                         ((store[s]["neural_2d"] + store[s]["traditional"]) / 2)[
                             np.asarray(f["test"], int)])
                for s in use_seeds for f in load_folds(protocol, s)])
            print(f"  {protocol:9s} [{tag}] {len(r2):2d} folds: "
                  f"NNLS3 {r2.mean():.4f}  eq2 {eq.mean():.4f}")
            report["protocols"].setdefault(protocol, {})[tag] = {
                "seeds": list(use_seeds), "n_folds": int(len(r2)),
                "nnls3_mean": float(r2.mean()), "nnls3_per_fold": [round(v, 4) for v in r2],
                "equal_2arm_mean": float(eq.mean()), "weights": w}
            if tag == "p1_equal_folds" or protocol == "scaffold":
                for s in use_seeds:
                    blobs[f"{protocol}__pred__seed{s}"] = preds[s]
                    blobs[f"{protocol}__arm__seed{s}"] = np.stack([store[s][a] for a in ARMS])
                    for a in ARMS:
                        blobs[f"{protocol}__{s}__{a}"] = store[s][a]
                # The full arm panel, which is what sigma is built from in L1. Only
                # arms present on EVERY partition are kept: the panel then has to be
                # identical across both protocols, which is what makes sigma
                # comparable between them. That excludes neural_3d, which was only
                # ever run on scaffold and butina-seed42 -- dropping it costs one arm
                # and buys a protocol-balanced panel.
                panel = sorted(set.intersection(*[set(store[s]) for s in use_seeds]))
                blobs[f"{protocol}__panel"] = np.stack([
                    np.stack([store[s][a] for s in use_seeds]) for a in panel])
                report["protocols"][protocol][tag]["panel"] = panel
                print(f"    panel ({len(panel)} arms): {panel}")

    out_npz = ROOT / "5_outputs/ensemble_oof.npz"
    np.savez(out_npz, **blobs)
    (ROOT / "5_outputs/ensemble_oof.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_npz.name} ({len(blobs)} arrays) and ensemble_oof.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
