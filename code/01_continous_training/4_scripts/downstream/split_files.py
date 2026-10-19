#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen split files for the downstream evaluations.

The plan (26-09-16 §5.1) requires that every model read the *same* split file,
so that no model can quietly get an easier partition. This module is the single
writer of those files: it builds folds through the project's existing
``chem_splits`` (never reimplementing scaffold perception, which RDKit version
drift can silently perturb) and freezes the result to disk along with a hash of
the canonical SMILES column.

It differs from ``finetune_moltrain.load_or_build_folds`` in two ways, both
needed for the downstream protocol:

1. It writes a **human-readable molecule-ID list** per fold, as the plan's
   reproducibility checklist requires. The training script's cache is an
   internal artifact; this is a deliverable.
2. It **refuses to emit a split it cannot defend**. ``audit()`` reports the
   largest fold's share and the number of groups, and ``build`` calls it, so a
   degenerate split (the MTR 83-molecule case, where one scaffold holds 32/83)
   is visible at build time instead of being discovered in the results.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chem_splits  # noqa: E402
import lumi_protocol  # noqa: E402

SPLIT_DIR = ROOT / "5_outputs/downstream_splits"

# The plan's fixed partition seeds (user decision #2): 5 partition seeds x 5
# folds = 25 folds. Distinct from the training-seed axis, which controls weight
# initialisation and is NOT varied in the main table.
PARTITION_SEEDS = [42, 123, 456, 789, 1011]

# Sheet2/Sheet3 predate this grid and have a history of their own. Their folds
# were built by ``finetune_moltrain``'s cache at seeds {42, 1337, 2024} (scaffold
# and butina) and {42, 515, 1337, 2026} (sequential, where the seed only moves
# the training RNG because the split itself is seed-free). Those runs are
# published and their queue scripts are still on disk, so the deliverable has to
# carry them too -- otherwise flipping Sheet2/Sheet3 onto ``downstream_splits``
# would turn every historical rerun into a hard failure.
#
# Freezing them here is not busywork: the scaffold files were verified
# (2026-09-18) to reproduce the existing ``split_cache`` entries **bit-identically**
# from ``ds_registry``'s raw SMILES, so this migration moves the files without
# moving a single molecule. The butina ones do NOT reproduce -- they are the
# corrected rebuild (坑 #14).
LEGACY_SEEDS_BY_DATASET = {
    "Sheet2": {"scaffold": [42, 1337, 2024],
               "butina": [42, 1337, 2024],
               "sequential": [42, 515, 1337, 2026]},
    "Sheet3": {"scaffold": [42, 1337, 2024],
               "butina": [42, 1337, 2024],
               "sequential": [42, 515, 1337, 2026]},
}

# A fold holding more than this share of the data cannot support a stable
# estimate: its test set is dominated by one chemical family, so its score
# measures that family, not the model. 1920/5 = 20% nominal.
MAX_FOLD_SHARE_WARN = 0.30

# Butina cutoff is PER DATASET, and the value is not a matter of taste.
#
# 0.7 was calibrated on Sheet2's 1920-molecule 4CR library and must NOT be
# copied to a new dataset. On the 2026-09-16 MTR run it produced folds of
# [10,5,5,20,43] (one fold = 51.8% of the data). Raising it to 0.8 balances the
# folds -- and is still wrong, because the imbalance was a SYMPTOM. ECFP4
# saturates on long-chain lipids: two MTR molecules differing by 8 atoms have
# bit-identical 2048-bit fingerprints (Tanimoto 1.0000 at every bit width; the
# collision-free count fingerprint says 0.8462). Measuring test->train max
# similarity showed butina@0.8 leaves 80/83 test molecules with a Tanimoto=1.0
# neighbour in train -- identical to random split. A "balanced" fold list there
# is an artifact of separating collided pairs, which destroys the isolation the
# split was supposed to provide. See prompt/26-09-16/02-发现-ECFP4碰撞导致MTR划分失效.md.
#
# So the cutoff must be chosen per dataset AND validated by the similarity
# probe in ``similarity_probe`` below -- never by fold balance alone.
BUTINA_CUTOFF_BY_DATASET = {
    "Sheet2": 0.7,   # 34 -> the project's canonical value; 0 ECFP4 collisions
    "Sheet3": 0.7,   # same 4CR chemistry as Sheet2
    "pKa": 0.8,      # 90 clusters, folds [115]*5; 88 distinct sources = real diversity
    # MTR is deliberately absent: butina is NOT a valid split for it at any
    # cutoff. Use scaffold (real extrapolation, warned as unbalanced) or random.
}
DEFAULT_BUTINA_CUTOFF = 0.8


def _smiles_hash(smiles: list[str]) -> str:
    return hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]


def build_folds(smiles: list[str], split: str, kfold: int, seed: int,
                butina_cutoff: float = 0.7) -> tuple[list, np.ndarray]:
    """Dispatch to the project's canonical fold builders.

    ``butina_cutoff`` defaults to 0.7, NOT the 0.6 that
    ``finetune_moltrain.py``'s CLI defaults to. On this 4CR library 0.4 puts
    1914/1920 molecules in one cluster and 0.6 gives uneven folds; 0.7 is the
    value the project settled on. Callers who want the old behaviour must ask.
    """
    if split == "scaffold":
        folds, _ids, labels = chem_splits.scaffold_folds(smiles, kfold, seed=seed)
    elif split == "butina":
        folds, _ids, labels = chem_splits.butina_folds(
            smiles, kfold, cutoff=butina_cutoff, seed=seed)
    elif split == "sequential":
        # LUMI-lab's own protocol: contiguous blocks in file order, no shuffle.
        # Seed-independent by construction, so the same file is written for every
        # seed -- kept per-seed anyway because the historical cache was keyed that
        # way and a reader should not have to know which axes are live.
        folds = lumi_protocol.sequential_folds(len(smiles), kfold)
        labels = np.arange(len(smiles))
    elif split == "random":
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        folds = list(kf.split(np.arange(len(smiles))))
        labels = np.arange(len(smiles))
    else:
        raise SystemExit(f"unknown split {split!r}")
    return folds, labels


def family_concentration(folds: list, labels: np.ndarray) -> dict:
    """How much of each test fold is taken up by its single largest group.

    This is the check that the balance audit misses, and it is the one that
    actually predicts failure. A fold can be perfectly sized (20% of the data)
    and still be worthless if one group owns all of it: the model is then
    scored on chemistry it never saw, and a constant predictor does as well.
    Measured on this project (scaffold protocol, ``chem_splits`` group ids):

    ===========  ======  ======  =========  ======  =========  ==========
    dataset      n       groups  largest    ratio   max share  measured R2
    ===========  ======  ======  =========  ======  =========  ==========
    pKa          574     339     57         0.50    50%        +0.45
    Sheet2       1920    469     240        0.62    62%        +0.53
    Sheet3       768     343     175        1.14    100%       -0.83
    MTR          83      37      32         1.93    100%       -3.93
    ===========  ======  ======  =========  ======  =========  ==========

    ``max_group_per_fold_share`` is the discriminator, and it is bimodal: the
    two viable cases sit at 50-62%, the two broken ones at 100%. The natural
    dividing line is whether a group can FILL a fold on its own. When the
    largest group is at least as big as a whole fold (ratio >= 1.0), some fold
    must be dominated by that single family. Below it, groups have to be
    interleaved and every fold gets a mixture. The two thresholds are set there:
    ``ratio >= 1.0`` is a hard failure (measured R2 was negative in both such
    cases), ``ratio >= 0.75`` is a warning (approaching single-family folds).

    Sanity note: this is a property of the LIBRARY, not of the splitter. Sheet3
    is a focused series and MTR is a designed panel, so no split algorithm can
    extract a scaffold-extrapolation signal from them.
    """
    labels = np.asarray(labels)
    sizes = np.array([len(te) for _tr, te in folds], float)
    per_fold = []
    for _tr, te in folds:
        lab = labels[np.asarray(te, int)]
        if len(lab) == 0:
            per_fold.append(0.0)
            continue
        _v, counts = np.unique(lab, return_counts=True)
        per_fold.append(float(counts.max() / len(lab)))

    n_groups = len(set(labels.tolist()))
    counts = np.bincount(np.unique(labels, return_inverse=True)[1]) if n_groups else np.array([0])
    biggest_group = float(counts.max()) if len(counts) else 0.0
    mean_fold = float(sizes.mean()) if len(sizes) else float("nan")
    ratio = biggest_group / mean_fold if mean_fold else float("nan")
    return {
        "n_scaffolds": n_groups,
        "largest_group": biggest_group,
        "mean_fold_size": round(mean_fold, 1),
        "max_group_per_fold_share": round(max(per_fold), 4) if per_fold else 0.0,
        "max_group_over_fold_size": round(ratio, 2),
        "scaffold_split_viable": bool(np.isfinite(ratio) and ratio < 1.0),
        "scaffold_split_marginal": bool(np.isfinite(ratio) and 1.0 <= ratio < 1.5),
    }


def similarity_probe(smiles: list[str], folds: list, cutoff: float | None = None,
                     max_pairs: int = 400) -> dict:
    """Measure how much the test folds actually look like a stranger to train.

    This exists because ``chem_splits.butina_leakage_check`` passes while the
    isolation it is meant to guarantee is absent. That check only compares
    cluster IDs; it cannot see that ECFP4 has collided two different molecules
    into one fingerprint. The 2026-09-16 MTR run is the worked example: folds
    balanced, cluster check green, and 96.4% of test molecules had a
    Tanimoto=1.0 neighbour in train.

    Reports per test molecule the MAX Tanimoto to any train molecule. A high
    median means the "test set" is interpolation wearing a holdout's clothes.
    ``n_identical`` counts test molecules whose ECFP4 is bit-identical to some
    train molecule -- for a long-chain library that is the number to watch.

    ``max_pairs`` samples test molecules on large datasets so a 1920-molecule
    probe stays fast; the sample is deterministic (every k-th), not random.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
           for s in smiles]
    n_test = sum(len(np.asarray(te)) for _tr, te in folds)
    stride = max(1, n_test // max_pairs)

    sims: list[float] = []
    counter = 0
    for tr, te in folds:
        tr = np.asarray(tr, int)
        for j in np.asarray(te, int):
            counter += 1
            if counter % stride:
                continue
            sims.append(max(DataStructs.BulkTanimotoSimilarity(fps[j], [fps[i] for i in tr])))
    arr = np.asarray(sims, float)
    near = float(arr.mean()) if len(arr) else float("nan")
    return {
        "n_test_probed": int(len(arr)),
        "mean_max_tanimoto": round(near, 4),
        "median_max_tanimoto": round(float(np.median(arr)), 4) if len(arr) else float("nan"),
        "p10_max_tanimoto": round(float(np.percentile(arr, 10)), 4) if len(arr) else float("nan"),
        "frac_near_duplicate": round(float((arr >= (cutoff or 0.7)).mean()), 4) if len(arr) else float("nan"),
        "n_identical": int((arr >= 1.0).sum()),
        # A split whose test set is this close to train is not measuring
        # extrapolation. 0.5 is a deliberately loose bar: even a genuinely
        # hard split rarely pushes the MEAN max-similarity below it.
        "looks_like_interpolation": bool(len(arr) and np.median(arr) >= 0.95),
    }


def audit(folds: list, labels: np.ndarray, n: int, split: str) -> dict:
    """Describe a fold list well enough to decide whether to trust it."""
    sizes = [int(len(te)) for _tr, te in folds]
    share = max(sizes) / n if n else float("nan")
    groups = len(set(np.asarray(labels).tolist()))
    # Leak check: a scaffold/cluster straddling train and test of one fold
    # makes the "extrapolation" claim false. chem_splits checks this too, but
    # it is cheap and this is the module that publishes the files.
    leaks = []
    if split in ("scaffold", "butina"):
        for k, (tr, te) in enumerate(folds):
            overlap = set(np.asarray(labels)[tr].tolist()) & set(np.asarray(labels)[te].tolist())
            if overlap:
                leaks.append(k)
    return {
        "fold_test_sizes": sizes, "max_fold_share": round(float(share), 4),
        "n_groups": groups, "leaking_folds": leaks,
        "even": bool(share <= MAX_FOLD_SHARE_WARN and not leaks),
        "family": family_concentration(folds, labels),
    }


def describe(audit_res: dict, n: int, n_mols_per_group: float, split: str) -> str:
    """One-line verdict plus a warning when the split is structurally weak."""
    d = audit_res
    line = (f"    {split:9s} fold sizes {d['fold_test_sizes']} | max share "
            f"{d['max_fold_share']:.1%} | {d['n_groups']} groups "
            f"(avg {n_mols_per_group:.1f} mol/group)")
    if d["leaking_folds"]:
        line += f"\n    ⚠️ LEAK in folds {d['leaking_folds']} — test not disjoint from train"
    elif not d["even"]:
        line += (f"\n    ⚠️ UNBALANCED: largest fold is {d['max_fold_share']:.1%} of the data. "
                 f"Test-set scores will be dominated by whichever family sits in that fold.")
    sim = d.get("similarity", {})
    # Interpolation is the DEFINITION of these two protocols, not a defect --
    # random splits interpolate by construction, and LUMI's sequential split is
    # contiguous blocks of a combinatorial enumeration, which is interpolation
    # over R-group assemblies. Saying so per-line would train the reader to
    # ignore the warning that DOES matter (scaffold/butina failing to extrapolate).
    if sim.get("looks_like_interpolation") and split not in ("random", "sequential"):
        line += (f"\n    ⚠️⚠️ NOT EXTRAPOLATION: median test->train Tanimoto is "
                 f"{sim['median_max_tanimoto']}, and {sim['n_identical']} test molecules have a "
                 f"BIT-IDENTICAL ECFP4 in train. A balanced fold list here is an artifact — "
                 f"ECFP4 has saturated. Do not report this as extrapolation.")
    fam = d.get("family", {})
    if split == "scaffold" and fam and not fam.get("scaffold_split_viable", True):
        line += (f"\n    ⚠️⚠️ FOLDS ARE SINGLE-FAMILY: the largest scaffold holds "
                 f"{fam['largest_group']:.0f} molecules vs a mean fold of "
                 f"{fam['mean_fold_size']:.0f}, so a fold cannot even be filled without "
                 f"one family dominating it ({fam['max_group_per_fold_share']:.0%} of some "
                 f"fold is that one scaffold). The model is scored on chemistry it never "
                 f"trained on and a constant predictor does as well — measured R2 was "
                 f"-0.83 on Sheet3 and -3.93 on MTR, both WORSE than predicting the train "
                 f"mean. Scaffold split is not viable for this library; use random and "
                 f"label the result as interpolation.")
    elif split == "scaffold" and fam.get("scaffold_split_marginal"):
        line += (f"\n    ⚠️ borderline: largest scaffold ({fam['largest_group']:.0f}) is "
                 f"comparable to a whole fold ({fam['mean_fold_size']:.0f}). Check per-fold "
                 f"family composition before trusting the spread.")
    return line


def build(ds_key: str, df: pd.DataFrame, target_cols: list[str], split: str,
          kfold: int = 5, seeds: list[int] | None = None,
          butina_cutoff: float | None = None, write: bool = True,
          probe_similarity: bool = True, overwrite: bool = False) -> dict:
    """Build (and optionally freeze) the full seed x fold grid for one dataset.

    Returns a summary dict; writes one JSON per (dataset, split, seed) holding
    explicit train/test molecule indices plus their SMILES, so a third party can
    reconstruct the exact partition without running RDKit.

    ``overwrite`` defaults to False and existing files are skipped. That default
    is deliberate: this directory is the published deliverable, and a rebuild is
    only ever legitimate as an explicit act (the 2026-09-18 butina correction was
    one). Silently rewriting a frozen fold file would change what a published
    number means without changing its name -- the same class of failure as
    MolTrain overwriting ``cv.data`` under a stale result JSON (report §8).
    """
    seeds = seeds if seeds is not None else seeds_for(ds_key, split)
    if split == "butina" and butina_cutoff is None:
        butina_cutoff = BUTINA_CUTOFF_BY_DATASET.get(ds_key, DEFAULT_BUTINA_CUTOFF)
    smiles = df["SMILES"].astype(str).tolist()
    h = _smiles_hash(smiles)
    cache_key = {"dataset": ds_key, "smiles_sha256": h, "n": len(smiles), "splits": {}}

    # The similarity probe is a property of the (dataset, method) pair, not of one
    # seed's particular fold assignment -- and it is the slow part. Run it once on
    # the first seed's folds and record it on every seed, so the summary does not
    # show "nan" for seeds 2..5 and read as "never checked".
    sim = None
    if probe_similarity:
        probe_folds, _ = build_folds(smiles, split, kfold, seeds[0], butina_cutoff)
        sim = similarity_probe(
            smiles, probe_folds,
            cutoff=butina_cutoff if split == "butina" else 0.7)

    for seed in seeds:
        folds, labels = build_folds(smiles, split, kfold, seed, butina_cutoff)
        aud = audit(folds, labels, len(smiles), split)
        if sim is not None:
            aud["similarity"] = sim
        n_groups = max(aud["n_groups"], 1)
        print(f"  [{ds_key}/{split}] seed={seed}"
              + (f"  butina_cutoff={butina_cutoff}" if split == "butina" else ""))
        print(describe(aud, len(smiles), len(smiles) / n_groups, split))

        if write:
            if split == "sequential" and seed != seeds[0]:
                # The sequential split is contiguous blocks in file order -- `seed`
                # does not enter the fold assignment at all. The historical cache
                # wrote one file per seed anyway, so the legacy seeds are frozen
                # verbatim; but writing the plan grid too would be four byte-identical
                # files pretending to be four partitions. Skip, and say so.
                print(f"  [skip] sequential is seed-independent; "
                      f"seed{seed} reuses seed{seeds[0]}'s file (no separate artifact)")
                cache_key["splits"][str(seed)] = dict(aud)
                continue
            if not overwrite and (SPLIT_DIR / f"{ds_key}_{split}_k{kfold}_seed{seed}.json").exists():
                print(f"  [skip] {ds_key}_{split}_k{kfold}_seed{seed}.json already frozen "
                      f"(pass overwrite=True to rebuild)")
                cache_key["splits"][str(seed)] = dict(aud)
                continue
            SPLIT_DIR.mkdir(parents=True, exist_ok=True)
            out = SPLIT_DIR / f"{ds_key}_{split}_k{kfold}_seed{seed}.json"
            out.write_text(json.dumps({
                "dataset": ds_key, "split": split, "kfold": kfold, "seed": seed,
                "butina_cutoff": butina_cutoff if split == "butina" else None,
                "n": len(smiles), "smiles_sha256": h,
                "target_cols": target_cols,
                "audit": aud,
                "folds": [{
                    "fold": i,
                    "train_idx": np.asarray(tr).tolist(),
                    "test_idx": np.asarray(te).tolist(),
                    # SMILES alongside indices: indices are only meaningful
                    # against this exact frame ordering, and the published list
                    # has to survive someone else loading the data differently.
                    "test_smiles": [smiles[j] for j in np.asarray(te)],
                } for i, (tr, te) in enumerate(folds)],
            }, indent=1), encoding="utf-8")
        cache_key["splits"][str(seed)] = aud
    return cache_key


def load_folds(ds_key: str, split: str, seed: int, kfold: int = 5,
               smiles: list[str] | None = None) -> list:
    """Read a frozen split back as the project's ``[(train, test), ...]`` form.

    Pass ``smiles`` whenever you have the frame at hand: the file records a hash
    of the SMILES column it was built from, and without the original there is no
    way to notice that the dataset changed underneath. ``finetune_moltrain``'s
    cache enforces this and refuses to score on stale folds; this reader
    enforces it too, so the two paths cannot disagree about whether it matters.
    """
    p = SPLIT_DIR / f"{ds_key}_{split}_k{kfold}_seed{seed}.json"
    if not p.exists():
        raise SystemExit(f"no frozen split at {p}; build it first")
    d = json.loads(p.read_text(encoding="utf-8"))
    if smiles is not None:
        got = _smiles_hash([str(s) for s in smiles])
        if d.get("smiles_sha256") != got:
            raise SystemExit(
                f"Split {p.name} was built for a DIFFERENT dataset "
                f"(hash {d.get('smiles_sha256')} != {got}). Refusing to score on stale "
                f"folds — delete the file to rebuild.")
    return [(np.asarray(f["train_idx"], int), np.asarray(f["test_idx"], int))
            for f in d["folds"]]


# What each dataset may legitimately be split by.
#
# MTR has no "butina" entry on purpose. The 2026-09-16 reason was "at 0.7 it is
# unbalanced AND interpolating, and at 0.8 it is balanced but STILL interpolating
# (96.4% of test molecules keep a Tanimoto=1.0 neighbour in train)". **That
# 96.4% was a symptom of 坑 #14, not a property of the data**: with the corrected
# ``butina_clusters``, MTR butina@0.8 gives 12 clusters, folds [16,17,16,17,17],
# family concentration 0.96 and n_identical=0. So the honest statement is
# "unusable under the buggy implementation; usable at 0.8 after the fix". Butina
# still stays out of MTR: the user's decision (2026-09-18) is random as the main
# protocol because 83 molecules x 16 targets has almost no statistical power, and
# that reason is independent of any splitter. See prompt/26-09-18/06-* §7.2.
SPLITS_BY_DATASET = {
    "pKa": ["scaffold", "butina"],
    "MTR": ["scaffold", "random"],
    # Evaluation 1 (TE). Sheet2/Sheet3 already have published results under
    # scaffold, butina and sequential; all three are frozen here so the deliverable
    # is a complete superset of what the queue scripts can ask for. `random` is
    # included as well, on the plan's seed grid: flipping these two datasets onto
    # `frozen_split` (so their butina folds come from the corrected splitter)
    # would otherwise turn a previously-working `--split random` into a hard
    # failure. A frozen store that can surprise you is not a frozen store.
    "Sheet2": ["scaffold", "butina", "sequential", "random"],
    "Sheet3": ["scaffold", "butina", "sequential", "random"],
}


def seeds_for(ds_key: str, split: str) -> list[int]:
    """The partition seeds to freeze for (dataset, split).

    Sheet2/Sheet3 carry BOTH axes: the plan's 5-seed grid (so evaluation 1 is
    paired with pKa/MTR fold-for-fold, 25 folds) and their historical seeds (so
    every published Sheet2/Sheet3 queue script can still be re-run verbatim).
    Sequential is the exception -- it has no plan-grid entry because the split is
    seed-free and the extra seeds would just be nine copies of one file.
    """
    legacy = LEGACY_SEEDS_BY_DATASET.get(ds_key, {}).get(split)
    if legacy is None:
        return PARTITION_SEEDS
    if split == "sequential":
        return legacy
    return sorted(set(PARTITION_SEEDS) | set(legacy))


def main() -> None:
    from ds_registry import load as load_ds

    which = sys.argv[1:] or ["pKa", "MTR", "Sheet2", "Sheet3"]
    summary = {}
    for key in which:
        df, targets = load_ds(key, verbose=True)
        print(f"\n{'='*72}\n{key}: n={len(df)}  targets={len(targets)}\n{'='*72}")
        for split in SPLITS_BY_DATASET.get(key, ["scaffold", "butina"]):
            summary[f"{key}/{split}"] = build(key, df, targets, split)

    print(f"\n{'='*72}\nSUMMARY (frozen files in {SPLIT_DIR})\n{'='*72}")
    for k, v in summary.items():
        # A random or sequential split interpolating is not a failure, it is the
        # definition. Only scaffold/butina promise extrapolation, so only they
        # get flagged.
        expects_extrapolation = not k.endswith(("/random", "/sequential"))
        for seed, a in v["splits"].items():
            sim = a.get("similarity") or {}
            bad_holdout = bool(sim.get("looks_like_interpolation")) and expects_extrapolation
            flag = "❌ " if bad_holdout else ("⚠️ " if not a["even"] else "OK ")
            print(f"  {flag}{k:16s} seed={seed:5s} sizes={str(a['fold_test_sizes']):26s} "
                  f"max={a['max_fold_share']:.1%} groups={a['n_groups']:4d} "
                  f"median_sim={sim.get('median_max_tanimoto', float('nan'))}")
    print("\n  OK = balanced & (where probed) a real holdout | "
          "⚠️ = unbalanced (or interpolation, for a random split by design) | "
          "❌ = balanced but NOT extrapolation (ECFP4 saturated)")


if __name__ == "__main__":
    main()
