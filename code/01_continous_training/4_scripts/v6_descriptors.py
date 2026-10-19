#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v6_descriptors.py — v6 MTR target descriptors (paper-style full RDKit + lipid priors).

v6 moves from v5's 32 curated descriptors to:
  1. the FULL RDKit descriptor set (Descriptors.CalcMolDescriptors, ~200 cols),
     exactly as the reference paper ("Transformers for Molecular Property
     Prediction: Domain Adaptation ..." uses all RDKit descriptors);
  2. 3 lipid-prior descriptors that are *domain-specific* for ionizable lipids:
       - lip_tail_logp      : Crippen LogP summed over hydrophobic tail segments
                              (pure-aliphatic-carbon connected components, >=4 C)
       - lip_tert_amine_pka : 10.0 if an aliphatic tertiary amine is present
                              (literature pKa ~9.8-10.7), else 0.0
       - lip_ionizable_n    : number of aliphatic tertiary amines (ionizable N)

Column selection: names are kept iff they (a) are finite on >= (1-max_nan_ratio)
of a representative sample and (b) have std > min_std (constant cols carry no
information for regression). NaN/inf are computed by the same gate logic as v5.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

LIPID_PRIOR_NAMES = ["lip_tail_logp", "lip_tert_amine_pka", "lip_ionizable_n"]

# Aliphatic tertiary amine: trivalent N, no H, not amide (N-C=O), not N-O/S
# (nitro/sulfonyl), not attached to aromatic C, not enamine (N-C=C).
_TERT_AMINE_SMARTS = "[NX3;H0;!$(N-C=O);!$(N~[#8,#16]);!$(N-c);!$(N-[#6]=[#6])]"
_TERT_AMINE_PAT = Chem.MolFromSmarts(_TERT_AMINE_SMARTS)

_TERT_AMINE_PKA = 10.0  # trimethylamine 9.8 / triethylamine 10.7


def _hydrophobic_tail_components(mol: Chem.Mol) -> list[list[int]]:
    """Pure-aliphatic-carbon connected components (C-C bonds only), size >= 4."""
    n = mol.GetNumAtoms()
    visited = [False] * n
    tails: list[list[int]] = []
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if visited[i]:
            continue
        if a.GetSymbol() != "C" or a.GetIsAromatic():
            continue
        comp: list[int] = []
        stack = [i]
        visited[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            for nb in mol.GetAtomWithIdx(j).GetNeighbors():
                k = nb.GetIdx()
                if not visited[k] and nb.GetSymbol() == "C" and not nb.GetIsAromatic():
                    visited[k] = True
                    stack.append(k)
        if len(comp) >= 4:
            tails.append(comp)
    return tails


def lip_tail_logp(mol: Chem.Mol) -> float:
    """Crippen LogP summed over hydrophobic tail segments.

    Crippen atom contributions need explicit H (they include H contribution),
    so we AddHs once and index contributions by the heavy-atom idx.
    """
    contribs = rdMolDescriptors._CalcCrippenContribs(
        Chem.AddHs(mol)
    )  # list of (logp_contrib, mr_contrib) in atom-idx order (H appended last)
    total = 0.0
    for comp in _hydrophobic_tail_components(mol):
        total += sum(contribs[i][0] for i in comp)
    return float(total)


def lip_tert_amine_pka(mol: Chem.Mol) -> float:
    """10.0 if an aliphatic tertiary amine is present, else 0.0."""
    if _TERT_AMINE_PAT is None:
        return 0.0
    return float(_TERT_AMINE_PKA if mol.HasSubstructMatch(_TERT_AMINE_PAT) else 0.0)


def lip_ionizable_n(mol: Chem.Mol) -> float:
    """Number of aliphatic tertiary amines (ionizable nitrogens)."""
    if _TERT_AMINE_PAT is None:
        return 0.0
    return float(len(mol.GetSubstructMatches(_TERT_AMINE_PAT)))


def compute_lipid_priors(mol: Chem.Mol) -> dict[str, float]:
    return {
        "lip_tail_logp": lip_tail_logp(mol),
        "lip_tert_amine_pka": lip_tert_amine_pka(mol),
        "lip_ionizable_n": lip_ionizable_n(mol),
    }


def all_rdkit_names() -> list[str]:
    """All descriptor names in RDKit's global registry."""
    return [name for name, _ in Descriptors._descList]


def compute_descriptors(smiles: str, desc_names: list[str]) -> np.ndarray:
    """Full-RDKit + lipid-prior descriptor vector; NaN where uncomputable.

    Mirrors build_v5_dataset.compute_descriptors's contract: float32 vector,
    NaN on parse failure. Uses CalcMolDescriptors once (fast C loop) for the
    RDKit block, plus the 3 lipid priors appended in LIPID_PRIOR_NAMES order.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(len(desc_names), float("nan"), dtype=np.float32)

    n_rdkit = len(desc_names) - len(LIPID_PRIOR_NAMES)
    vals = np.empty(len(desc_names), dtype=np.float64)
    try:
        all_desc = Descriptors.CalcMolDescriptors(mol)
    except Exception:
        return np.full(len(desc_names), float("nan"), dtype=np.float32)

    for i, name in enumerate(desc_names[:n_rdkit]):
        try:
            v = all_desc.get(name, float("nan"))
            vals[i] = v if v is not None else float("nan")
        except Exception:
            vals[i] = float("nan")

    priors = compute_lipid_priors(mol)
    for j, name in enumerate(LIPID_PRIOR_NAMES):
        vals[n_rdkit + j] = priors[name]
    return vals.astype(np.float32)


def select_desc_names(
    smiles_list: list[str],
    max_nan_ratio: float = 0.005,
    min_std: float = 1e-8,
) -> list[str]:
    """Keep descriptor columns that are (nearly) always finite and non-constant."""
    cand = all_rdkit_names() + LIPID_PRIOR_NAMES
    if not smiles_list:
        return cand

    n = min(len(smiles_list), 5000)
    mat = np.empty((n, len(cand)), dtype=np.float64)
    for row, smi in enumerate(smiles_list[:n]):
        mat[row] = compute_descriptors(smi, cand)
    return select_desc_names_from_matrix(mat)


def select_desc_names_from_matrix(
    mat: np.ndarray,
    max_nan_ratio: float = 0.005,
    min_std: float = 1e-8,
) -> list[str]:
    """Column selection from a precomputed [N, all_desc] sample matrix.

    Names are kept iff they are finite on >= (1-max_nan_ratio) of the sample
    and have std > min_std (constant columns carry no regression signal).
    """
    cand = all_rdkit_names() + LIPID_PRIOR_NAMES
    finite = np.isfinite(mat)
    keep = []
    for col, name in enumerate(cand):
        nan_ratio = 1.0 - finite[:, col].mean()
        if nan_ratio > max_nan_ratio:
            continue
        vals = mat[finite[:, col], col]
        if vals.size < 2 or vals.std() < min_std:
            continue
        keep.append(name)
    return keep
