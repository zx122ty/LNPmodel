#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conformer_3d.py — 3D conformer generation for DOWNSTREAM fine-tuning.

Why this module exists
----------------------
MolTrain's built-in conformer path is ``inner_smi2coords(mode='fast')``
(unimol_tools/data/conformer.py:263-280):

    res = AllChem.EmbedMolecule(mol, randomSeed=seed)   # default maxAttempts=10
    if res == 0:
        ...                       # real 3D
    else:
        AllChem.Compute2DCoords(mol)   # <-- SILENT fallback to a FLAT conformer

These ionizable lipids are large and flexible (Sheet2: 38-69 heavy atoms, dozens of
rotatable bonds). ``EmbedMolecule`` with the default 10 attempts returns -1 on the
large majority of them, so the model is fine-tuned on planar coordinates while it
was PRETRAINED on real 3D (build_v5_dataset.py uses ETKDGv3 + MMFF). Measured on
the actual training input (``Sheet2_data.sdf``): 81.8% of rows have a z-extent
below 0.5 A. Uni-Mol2's attention consumes interatomic distances, so a flattened
conformer is not a small perturbation -- it is a different input distribution.

This module reproduces the PRETRAINING recipe exactly (ETKDGv3 + useRandomCoords +
bounded iteration budget + MMFF optimisation) and reports the 3D success rate so a
silent regression to 2D can never happen again.

The 3D-success gate
-------------------
A formally "embedded" conformer can still be nearly planar, so counting
``EmbedMultipleConfs`` return codes is not enough. We measure the actual z-extent
and assert the population is overwhelmingly 3D; if not, the fine-tune aborts rather
than silently reproducing the original bug.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# A conformer is counted as "3D" once its z-extent clears this. Deliberately far
# below the ~14 A a real lipid spans and far above the 0.00 A a 2D depiction has,
# so the statistic is unambiguous rather than threshold-tuned.
FLAT_Z_EXTENT = 0.5

# Pretraining defaults (see build_v5_dataset.py):
#   EMBED_PRE_ITERATIONS / EMBED_ITERATIONS / CONF_MMFF_MAXITER
EMBED_PRE_ITERATIONS = 3
EMBED_ITERATIONS = 200
CONF_MMFF_MAXITER = 2000

# Escalation ladder for DOWNSTREAM molecules. The pretraining pipeline stops at the
# first rung and discards the ~3.5% that fail, which is fine across 1M molecules. A
# benchmark has ~2000 rows and every one carries unique supervision, so a failure is
# worth paying for: embedding cost grows with the iteration budget, and these lipids
# reach 46 rotatable bonds, the expensive corner of distance-geometry embedding.
EMBED_ATTEMPTS = (EMBED_PRE_ITERATIONS, EMBED_ITERATIONS, 2000, 20000)

# Downstream molecules are far smaller than the 256-atom pretraining budget
# (Sheet2 max 69 heavy atoms; Sheet3 max 180). Pass this explicitly to MolTrain:
# its UniMolV2Feature default is 128, and mol2unimolv2 crops by RANDOMLY SHUFFLING
# a boolean mask when the molecule exceeds the budget -> the 8 Sheet3 molecules over
# 128 atoms would silently lose a random subset of atoms.
MAX_ATOMS = 256


@dataclass
class EmbedStats:
    """Audit trail for one generation pass — the guard against silent 2D fallback."""

    n_total: int = 0
    n_3d: int = 0
    n_flat: int = 0
    n_failed: int = 0
    n_mmff: int = 0
    n_uff: int = 0
    n_unoptimised: int = 0
    z_extents: list[float] = field(default_factory=list)

    @property
    def rate_3d(self) -> float:
        return self.n_3d / self.n_total if self.n_total else 0.0

    def summary(self) -> str:
        med = float(np.median(self.z_extents)) if self.z_extents else 0.0
        return (f"3D={self.n_3d}/{self.n_total} ({100 * self.rate_3d:.1f}%), "
                f"flat={self.n_flat}, failed={self.n_failed}, "
                f"MMFF94s={self.n_mmff}/UFF={self.n_uff}/raw={self.n_unoptimised}, "
                f"median z-extent={med:.2f} A")


def optimise_conformer(mol: Chem.Mol) -> str:
    """MMFF94s with UFF fallback — byte-for-byte the pretraining recipe.

    ``prepare_lipids_unimol_dataset.optimize_conformer`` is the reference. Matching
    it matters: the pretrained weights saw MMFF94s-minimised geometries, so a
    fine-tune that minimises with a different force field feeds the encoder a
    subtly different geometry distribution. Returns the method actually used.
    """
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            status = AllChem.MMFFOptimizeMolecule(
                mol, mmffVariant="MMFF94s", maxIters=CONF_MMFF_MAXITER
            )
            if status == 0:
                return "MMFF94s"
    except Exception:
        pass

    try:
        if AllChem.UFFOptimizeMolecule(mol, maxIters=CONF_MMFF_MAXITER) == 0:
            return "UFF"
    except Exception:
        pass

    # Neither force field converged. The ETKDG geometry is still genuine 3D (that is
    # the property this module exists to guarantee), so keep it rather than fail.
    return "raw"


def embed_3d(smiles: str, seed: int = 42,
             attempts: tuple[int, ...] | None = None) -> tuple[Chem.Mol | None, float, str]:
    """One 3D conformer for one SMILES, using the pretraining recipe.

    Returns ``(mol_with_Hs, z_extent, optimiser)``. ``mol`` is None only when the
    molecule cannot be embedded at all — a failure is returned as a failure and is
    never silently replaced by a 2D depiction.

    The search order mirrors build_v5_dataset.prepare_v5_molecule:
      1. ETKDGv3 with useRandomCoords=True and a tiny iteration budget (long
         flexible chains need random initialisation to embed at all);
      2. if that fails, ETKDGv3 with default initialisation and the real budget
         (heads with aromatic/charged groups sometimes embed better without it).

    ``attempts`` is an escalation ladder of iteration budgets. The pretraining code
    stops at the first rung because it has 1M molecules and can afford to discard
    ~3.5% of them; a downstream benchmark has ~2000 rows where every molecule carries
    unique supervision, so it is worth paying for a much larger budget before giving
    up on one. The top rung is deliberately generous: these lipids reach 46 rotatable
    bonds, and distance-geometry embedding on that many degrees of freedom is the
    expensive corner of ETKDG.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 0.0, "parse_failed"
    mol = Chem.AddHs(mol)

    if attempts is None:
        attempts = EMBED_ATTEMPTS

    for max_iters in attempts:
        for use_random in (True, False):
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.enforceChirality = True
            params.useSmallRingTorsions = True
            params.useMacrocycleTorsions = True
            params.pruneRmsThresh = -1.0
            params.numThreads = 1
            params.useRandomCoords = use_random
            params.maxIterations = max_iters
            try:
                conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params))
            except Exception:
                conf_ids = []
            if conf_ids:
                optimiser = optimise_conformer(mol)
                coords = mol.GetConformer().GetPositions()
                z_extent = float(coords[:, 2].max() - coords[:, 2].min())
                return mol, z_extent, optimiser

    return None, 0.0, "embed_failed"


def _embed_task(task: tuple[str, int]) -> tuple[int, object | None, float, str]:
    """Worker: embed one molecule and return an RDKit mol carrying BOTH bond orders
    and the 3D conformer.

    Returning a mol rather than (atoms, coordinates) is the whole point. Passing bare
    atoms+coordinates to MolTrain routes through create_mol_from_atoms_and_coords
    (unimol_tools/data/conformer.py:511), which builds an RWMol with NO BONDS — so
    get_graph yields degree 0, empty edge_attr and shortest_path 511 (its
    "unreachable" sentinel) for every atom pair. Measured: that variant loses the
    entire 2D bond graph (shortest_path 1892/1936 entries differ, degree differs on
    every atom), which is why feeding it produced a large REGRESSION. A mol from
    SMILES keeps the real connectivity; only its conformer is replaced with the 3D
    geometry, which is the actual controlled variable.
    """
    smiles, seed = task
    mol, z_extent, optimiser = embed_3d(smiles, seed=seed)
    if mol is None:
        return -1, None, 0.0, optimiser
    return 0, mol, z_extent, optimiser


def generate_3d_conformers(
    smiles_list: list[str],
    seed: int = 42,
    n_workers: int | None = None,
    cache_path: Path | None = None,
    max_drop_frac: float = 0.02,
) -> tuple[list, list[int], EmbedStats]:
    """RDKit mols (correct bond orders + 3D conformer) for the embeddable molecules.

    Returns ``(mols, kept_indices, stats)``. ``kept_indices`` are positions into
    ``smiles_list``, so the caller can filter the target vector to match — a molecule
    that cannot be embedded must be DROPPED, never given a placeholder, because a
    fabricated conformer would poison its own supervision.

    Dropping rows is only safe because the caller re-slices its fold indices to the
    surviving rows (see ``remap_folds``). Refusing to run is reserved for a systemic
    failure (more than ``max_drop_frac`` of molecules), which means the pipeline is
    broken rather than unlucky — that is the "silent 2D fallback" class of bug this
    module exists to prevent.

    The returned mols keep explicit hydrogens: MolTrain's ``mol2unimolv2`` calls
    ``RemoveAllHs`` internally, so the H count must be right on the way in.
    """
    if n_workers is None:
        import os
        n_workers = max(1, min(32, (os.cpu_count() or 8) - 2))

    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        # SMILES are part of the cache key, not just the path: reusing a cache built
        # for a different molecule set would misalign atoms with targets.
        if cached.get("smiles_hash") == _smiles_hash(smiles_list):
            print(f"  [conf3d] cache hit -> {cache_path} "
                  f"({cached['stats'].summary()})", flush=True)
            return cached["mols"], cached["kept_indices"], cached["stats"]
        print(f"  [conf3d] cache SMILES mismatch, regenerating -> {cache_path}",
              flush=True)

    print(f"  [conf3d] generating 3D conformers for {len(smiles_list):,} molecules "
          f"({n_workers} workers, ETKDGv3 + MMFF, the pretraining recipe) …", flush=True)

    stats = EmbedStats(n_total=len(smiles_list))
    results: list[tuple[object | None, float, str]] = [None] * len(smiles_list)

    tasks = [(s, seed) for s in smiles_list]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, res in enumerate(ex.map(_embed_task, tasks, chunksize=4)):
            results[i] = res[1:]

    mols_out, kept = [], []
    dropped: list[tuple[int, str]] = []
    for i, (mol, z_extent, optimiser) in enumerate(results):
        if mol is None:
            dropped.append((i, optimiser))       # parse_failed / embed_failed
            stats.n_failed += 1
            continue
        if z_extent < FLAT_Z_EXTENT:
            dropped.append((i, "flat"))
            stats.n_flat += 1
            continue
        mols_out.append(mol)
        kept.append(i)
        stats.z_extents.append(z_extent)
        if optimiser == "MMFF94s":
            stats.n_mmff += 1
        elif optimiser == "UFF":
            stats.n_uff += 1
        else:
            stats.n_unoptimised += 1
        stats.n_3d += 1

    print(f"  [conf3d] {stats.summary()}", flush=True)

    if dropped:
        print(f"  [conf3d] DROPPING {len(dropped)} molecules without a 3D conformer "
              f"(rows are removed and fold indices remapped, so no fabricated input "
              f"reaches the model):", flush=True)
        for i, why in dropped[:10]:
            print(f"      row {i} ({why}): {smiles_list[i][:70]}", flush=True)
    if stats.n_failed + stats.n_flat > max_drop_frac * stats.n_total:
        raise SystemExit(
            f"[conf3d] {stats.n_failed + stats.n_flat}/{stats.n_total} molecules "
            f"({stats.rate_3d:.1%} 3D) exceed the {max_drop_frac:.1%} drop budget. "
            f"This is a systemic failure, not bad luck — refusing to fine-tune."
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump({
                "smiles_hash": _smiles_hash(smiles_list),
                "mols": mols_out,
                "kept_indices": kept,
                "stats": stats,
            }, fh)
        print(f"  [conf3d] cached -> {cache_path}", flush=True)

    return mols_out, kept, stats


def remap_folds(folds: list, kept_indices: list[int]) -> list:
    """Re-index cached folds after rows were dropped, preserving train/test identity.

    ``folds`` hold positions into the ORIGINAL row order; the featurised arrays and
    target vector now cover only ``kept_indices``. Without this remap every index
    after the first dropped row would shift by one and the split would silently mix
    train and test molecules — a scaffold split would still look plausible, which is
    exactly why it has to be done here rather than trusted.

    Test rows that were dropped are simply removed from that fold. Train rows that
    were dropped are removed from training. A fold whose test set collapses is
    reported rather than silently scored.
    """
    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    kept_set = set(kept_indices)
    out = []
    for k, (tr, te) in enumerate(folds):
        tr_kept = [old_to_new[i] for i in np.asarray(tr, dtype=int) if i in kept_set]
        te_kept = [old_to_new[i] for i in np.asarray(te, dtype=int) if i in kept_set]
        n_tr_drop = len(np.asarray(tr, dtype=int)) - len(tr_kept)
        n_te_drop = len(np.asarray(te, dtype=int)) - len(te_kept)
        if n_tr_drop or n_te_drop:
            print(f"  [conf3d] fold {k}: remapped after drops "
                  f"(train {len(np.asarray(tr, dtype=int))}->{len(tr_kept)}, "
                  f"test {len(np.asarray(te, dtype=int))}->{len(te_kept)})", flush=True)
        if len(te_kept) < 2:
            raise SystemExit(
                f"[conf3d] fold {k} has {len(te_kept)} test rows after dropping "
                f"unembeddable molecules — too few to score. Fix embedding coverage."
            )
        out.append((np.asarray(tr_kept, dtype=int), np.asarray(te_kept, dtype=int)))
    return out



def _smiles_hash(smiles_list: list[str]) -> str:
    return hashlib.sha256("\n".join(smiles_list).encode()).hexdigest()[:16]


def default_cache_path(dataset: str, n: int) -> Path:
    """Cache 3D conformers outside the per-run experiment dir so every base reuses them.

    The ``v2`` tag marks the payload as RDKit mols. v1 cached bare (atoms, coordinates),
    which MolTrain turns into a BOND-FREE molecule — silently discarding the whole 2D
    graph. Bumping the name stops a v1 file from being picked up by the fixed code.
    """
    root = Path(__file__).resolve().parents[1] / "5_outputs/conf3d_cache"
    return root / f"{dataset}_n{n}_etkdgv3_mol.pkl"
