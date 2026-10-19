#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prove evaluation 1's pre-flight is actually done, in one command.

Why this exists
---------------
``prompt/26-09-18/05-pKa评估结果.md`` §9.2 listed four things to do before
evaluation 1 could start: rebuild the Sheet2/Sheet3 butina folds, add the plan's
seed grid, add Sheet2/Sheet3 loaders to ``ds_registry``, and re-check the CPU
budget. Three of those were finished on 2026-09-18 evening, but the document
still said "the correct versions do not exist yet" -- and a later reader would
have redone work that was already done, or worse, distrusted folds that are
correct.

A prose claim in a handover document goes stale silently. A command does not.
This script is that command: it re-derives the four claims from the artifacts on
disk and exits non-zero if any of them stops being true.

The four checks
---------------
1. **Folds exist** -- all 7 seeds for Sheet2/Sheet3 x {scaffold, butina}.
2. **Folds are reproducible** -- regenerating every butina fold from the current
   ``chem_splits.butina_folds`` reproduces the frozen file index-for-index.
   This is the check that would have caught 坑 #14 the first time: the frozen
   folds were internally consistent, balanced, and leak-free, and still wrong.
3. **The legacy mirror is in sync** -- ``split_cache/`` (which the neural arm
   and ~20 legacy analysis scripts read) is fold-for-fold identical to
   ``downstream_splits/``, with ``synced_from`` naming the source. The buggy
   versions are still present but carry a ``.BUGGY_20260918`` suffix.
4. **Row order matches the trainer** -- ``ds_registry.load`` returns the same
   rows, in the same order, as ``finetune_moltrain.load_data``. Frozen folds are
   integer indices, so if this drifts the folds silently select other molecules.

Usage:
    python 4_scripts/downstream/verify_eval1_preflight.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

import chem_splits  # noqa: E402
import ds_registry  # noqa: E402
import split_files  # noqa: E402

SPLITS_DIR = ROOT / "5_outputs/downstream_splits"
CACHE_DIR = ROOT / "5_outputs/split_cache"
DATASETS = ("Sheet2", "Sheet3")
PROTOCOLS = ("scaffold", "butina")
SEEDS = (42, 123, 456, 789, 1011, 1337, 2024)
BUGGY_SUFFIX = ".BUGGY_20260918.json"


def _fold_key_splits(doc: dict) -> list[tuple[list, list]]:
    """downstream_splits schema: folds are {fold, train_idx, test_idx, ...}."""
    return [(list(f["train_idx"]), list(f["test_idx"])) for f in doc["folds"]]


def _fold_key_cache(doc: dict) -> list[tuple[list, list]]:
    """split_cache schema: folds are {train, test} -- the legacy shape."""
    return [(list(f["train"]), list(f["test"])) for f in doc["folds"]]


def check_folds_present() -> list[str]:
    bad = []
    for ds in DATASETS:
        for split in PROTOCOLS:
            for seed in SEEDS:
                p = SPLITS_DIR / f"{ds}_{split}_k5_seed{seed}.json"
                if not p.exists():
                    bad.append(f"missing {p.name}")
    return bad


def check_folds_reproducible() -> list[str]:
    """Regenerate every butina fold and compare to the frozen file.

    Compared as sets, because fold membership is what matters and the cluster
    ordering inside a fold is an implementation detail of the balancer.
    """
    bad = []
    for ds in DATASETS:
        df, _ = ds_registry.load(ds, verbose=False)
        smiles = df["SMILES"].astype(str).tolist()
        cutoff = split_files.BUTINA_CUTOFF_BY_DATASET[ds]
        for seed in SEEDS:
            folds, _ids, _cl = chem_splits.butina_folds(smiles, 5, cutoff=cutoff,
                                                        seed=seed)
            frozen = _fold_key_splits(json.loads(
                (SPLITS_DIR / f"{ds}_butina_k5_seed{seed}.json").read_text()))
            for k, ((tr, te), (f_tr, f_te)) in enumerate(zip(folds, frozen)):
                if not (np.array_equal(np.sort(tr), np.sort(f_tr))
                        and np.array_equal(np.sort(te), np.sort(f_te))):
                    bad.append(f"{ds} seed{seed} fold{k} differs from the frozen file")
    return bad


def check_cache_in_sync() -> list[str]:
    bad = []
    for ds in DATASETS:
        for seed in SEEDS:
            live = _fold_key_splits(json.loads(
                (SPLITS_DIR / f"{ds}_butina_k5_seed{seed}.json").read_text()))
            cache_doc = json.loads(
                (CACHE_DIR / f"{ds}_butina_k5_seed{seed}.json").read_text())
            cache = _fold_key_cache(cache_doc)
            for k, ((l_tr, l_te), (c_tr, c_te)) in enumerate(zip(live, cache)):
                if not (np.array_equal(l_tr, c_tr) and np.array_equal(l_te, c_te)):
                    bad.append(f"{ds} seed{seed} fold{k}: split_cache != downstream_splits")
            if not cache_doc.get("synced_from"):
                bad.append(f"{ds} seed{seed}: split_cache file has no synced_from")
    return bad


def check_buggy_quarantined() -> list[str]:
    """Every butina fold file in split_cache is either in sync or marked BUGGY.

    A file that is neither is the failure mode this whole archive exists to
    prevent: a void partition sitting under an authoritative name.
    """
    bad = []
    for p in sorted(CACHE_DIR.glob("*butina*")):
        if p.name.endswith(BUGGY_SUFFIX):
            continue
        if not p.name.startswith(("Sheet2_", "Sheet3_")):
            continue
        seed = int(p.name.rsplit("seed", 1)[1].split(".")[0])
        if seed not in SEEDS:
            bad.append(f"{p.name}: unmarked butina file for an unexpected seed")
    return bad


def check_row_order() -> list[str]:
    res = ds_registry.verify_against_moltrain()
    bad = []
    for ds, r in res.items():
        if not r["smiles_identical"]:
            bad.append(f"{ds}: row order differs from finetune_moltrain")
        if r["max_abs_target_delta"] != 0.0:
            bad.append(f"{ds}: max |Δtarget| = {r['max_abs_target_delta']} (expected 0)")
    return bad


CHECKS = [
    ("1. Sheet2/Sheet3 folds present (7 seeds x 2 protocols)", check_folds_present),
    ("2. butina folds reproducible from current chem_splits", check_folds_reproducible),
    ("3. split_cache mirror in sync with downstream_splits", check_cache_in_sync),
    ("4. no unmarked butina file left in split_cache", check_buggy_quarantined),
    ("5. ds_registry row order == finetune_moltrain.load_data", check_row_order),
]


def main() -> None:
    failed = 0
    for name, fn in CHECKS:
        bad = fn()
        if bad:
            failed += 1
            print(f"FAIL  {name}")
            for b in bad[:10]:
                print(f"        {b}")
            if len(bad) > 10:
                print(f"        ... and {len(bad) - 10} more")
        else:
            print(f"ok    {name}")

    print()
    if failed:
        print(f"{failed}/{len(CHECKS)} check(s) FAILED -- evaluation 1 is not "
              f"cleared to start")
        raise SystemExit(1)
    print(f"ALL {len(CHECKS)} CHECKS PASSED -- evaluation 1 pre-flight is done")


if __name__ == "__main__":
    main()
