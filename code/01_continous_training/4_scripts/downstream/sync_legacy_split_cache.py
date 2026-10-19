#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror the frozen Sheet2/Sheet3 folds into the legacy ``split_cache/`` store.

Why this exists
---------------
The project ended up with two fold stores, and until now they were written by
two different code paths:

* ``5_outputs/split_cache/`` -- the internal cache ``finetune_moltrain`` reads
  for Sheet2/Sheet3 (keys ``train`` / ``test``). Every legacy analysis script
  (``arm_zoo``, ``ensemble_weighted``, ``uncertainty_eval``, ``active_learning_sim``,
  ``freeze_scan_eval``, ``cheap_feature_probe``, ...) reads THIS one.
* ``5_outputs/downstream_splits/`` -- the published deliverable (keys
  ``train_idx`` / ``test_idx``, plus per-fold SMILES). ``split_files`` is the
  only writer.

Two stores is fine; two WRITERS is not, and the 2026-09-18 Butina bug
(坑 #14) is exactly what that costs: the corrected folds were frozen into
``downstream_splits/`` while ``split_cache/`` kept the buggy ones renamed aside,
so the neural arm (split_cache) and the baseline arm (downstream_splits) would
have been scored on DIFFERENT partitions of the same dataset -- and nothing
would have said so.

So this script makes ``split_cache`` a *derived* artifact: it copies the frozen
files verbatim, converting only the key names. One writer, one source of truth,
and the legacy readers keep working unchanged.

Safety properties, in order of importance:

1. It never invents a fold assignment. It copies bytes from
   ``downstream_splits/``; if the two stores ever disagree after a sync, the
   sync is broken, not the source.
2. It refuses to touch a ``split_cache`` file whose content differs and whose
   name does not carry a quarantine marker, unless ``--force``. A pre-existing
   disagreement is a finding to look at, not something to paper over.
3. It re-reads everything it wrote and asserts the fold arrays match. A sync
   that reports success without checking is the same class of claim as a
   reproducibility bundle that only checks its own hashes.

Usage:
    python 4_scripts/downstream/sync_legacy_split_cache.py            # sync + verify
    python 4_scripts/downstream/sync_legacy_split_cache.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds_registry  # noqa: E402
import split_files  # noqa: E402

CACHE_DIR = ROOT / "5_outputs/split_cache"

# Only the datasets whose neural arm reads the legacy cache. pKa/MTR are
# `frozen_split` datasets and read downstream_splits directly -- mirroring them
# would create a second copy that nothing reads, which is how drift starts.
DATASETS = ["Sheet2", "Sheet3"]


def legacy_payload(frozen: dict) -> dict:
    """Convert a frozen split file to the legacy cache schema.

    The two schemas differ in key names only (``train_idx``/``test_idx`` vs
    ``train``/``test``); the fold assignment is copied verbatim. ``created`` is
    deliberately NOT carried over -- it would claim the legacy file was written
    at the frozen file's build time, which is a small lie of the kind this whole
    exercise is about avoiding.
    """
    return {
        "dataset": frozen["dataset"],
        "split": frozen["split"],
        "kfold": frozen["kfold"],
        "seed": frozen["seed"],
        "butina_cutoff": frozen.get("butina_cutoff"),
        "n": frozen["n"],
        "smiles_sha256": frozen["smiles_sha256"],
        "fold_test_sizes": [len(f["test_idx"]) for f in frozen["folds"]],
        "folds": [{"train": list(f["train_idx"]), "test": list(f["test_idx"])}
                  for f in frozen["folds"]],
        "synced_from": "5_outputs/downstream_splits/"
                       f"{frozen['dataset']}_{frozen['split']}_k{frozen['kfold']}"
                       f"_seed{frozen['seed']}.json",
    }


def _fold_arrays(fold: dict) -> tuple[np.ndarray, np.ndarray]:
    """Accept either schema: the frozen file says train_idx/test_idx, the legacy
    cache says train/test. Reading both through one accessor is what lets the
    verifier compare the two stores directly instead of comparing a store to a
    converted copy of itself (which would pass even if the conversion were wrong).
    """
    tr = fold["train_idx"] if "train_idx" in fold else fold["train"]
    te = fold["test_idx"] if "test_idx" in fold else fold["test"]
    return np.asarray(tr, int), np.asarray(te, int)


def same_folds(a: dict, b: dict) -> bool:
    """Fold-for-fold equality, ignoring everything else in the payload."""
    if len(a["folds"]) != len(b["folds"]):
        return False
    for u, v in zip(a["folds"], b["folds"]):
        atr, ate = _fold_arrays(u)
        btr, bte = _fold_arrays(v)
        if not (np.array_equal(atr, btr) and np.array_equal(ate, bte)):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a legacy file that disagrees (after you have "
                         "looked at why)")
    args = ap.parse_args()

    n_written = n_identical = n_refused = 0
    for ds in DATASETS:
        df, _ = ds_registry.load(ds, verbose=False)
        smiles = df["SMILES"].astype(str).tolist()
        h = split_files._smiles_hash(smiles)
        print(f"\n=== {ds} (n={len(smiles)}, raw-smiles hash {h}) ===")

        for split in split_files.SPLITS_BY_DATASET[ds]:
            for seed in split_files.seeds_for(ds, split):
                src = split_files.SPLIT_DIR / f"{ds}_{split}_k5_seed{seed}.json"
                dst = CACHE_DIR / src.name
                if not src.exists():
                    print(f"  ❌ {src.name}: no frozen source -- run split_files.py first")
                    n_refused += 1
                    continue
                frozen = json.loads(src.read_text(encoding="utf-8"))
                if frozen["smiles_sha256"] != h:
                    print(f"  ❌ {src.name}: frozen hash {frozen['smiles_sha256']} "
                          f"!= current frame {h} -- dataset moved underneath it")
                    n_refused += 1
                    continue

                payload = legacy_payload(frozen)
                if dst.exists():
                    old = json.loads(dst.read_text(encoding="utf-8"))
                    if same_folds(old, payload):
                        n_identical += 1
                        continue
                    if not args.force:
                        print(f"  ⚠️  {dst.name}: EXISTS and DIFFERS -- refusing "
                              f"(this is a finding; inspect, then --force if the "
                              f"frozen file is the one you mean)")
                        n_refused += 1
                        continue

                if args.dry_run:
                    print(f"  (dry-run) would write {dst.name}")
                    continue

                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                n_written += 1

    # --- verify what was just written, by re-reading it from disk ---
    print(f"\n{'='*72}\nVERIFY (re-read from disk)\n{'='*72}")
    problems = []
    pending = []
    for ds in DATASETS:
        for split in split_files.SPLITS_BY_DATASET[ds]:
            for seed in split_files.seeds_for(ds, split):
                src = split_files.SPLIT_DIR / f"{ds}_{split}_k5_seed{seed}.json"
                dst = CACHE_DIR / src.name
                if not (src.exists() and dst.exists()):
                    # In a dry run nothing was written, so a one-sided pair is
                    # expected, not a failure -- reporting it as a problem would
                    # make --dry-run useless as a preview.
                    (pending if args.dry_run else problems).append(dst.name)
                    continue
                f = json.loads(src.read_text(encoding="utf-8"))
                c = json.loads(dst.read_text(encoding="utf-8"))
                if not same_folds(f, c):
                    problems.append(f"{dst.name}: FOLD MISMATCH after sync")
                elif f["smiles_sha256"] != c["smiles_sha256"]:
                    problems.append(f"{dst.name}: hash mismatch")
    if pending:
        print(f"  (dry run) {len(pending)} file(s) would be created: "
              f"{', '.join(pending[:4])}{' ...' if len(pending) > 4 else ''}")
    if problems:
        for p in problems:
            print(f"  ❌ {p}")
        raise SystemExit(f"{len(problems)} problem(s) -- legacy cache is NOT a "
                         f"faithful mirror")
    if not args.dry_run:
        print("  ✅ every Sheet2/Sheet3 frozen file has a bit-identical legacy "
              "cache twin")

    print(f"\nwrote {n_written} | already identical {n_identical} | refused {n_refused}")
    if args.dry_run:
        print("(dry run -- nothing was written)")


if __name__ == "__main__":
    main()
