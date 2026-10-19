"""Custom LNP domain-specific descriptor backend.

Computes SMARTS-based structural motif features and handcrafted physicochemical
features specifically relevant to ionizable lipid nanoparticle SAR.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors

from .base import DescriptorBackend

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


def _count_smarts(mol: Mol, pattern: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(pattern))


def _has_smarts(mol: Mol, pattern: Chem.Mol) -> int:
    return int(mol.HasSubstructMatch(pattern))


def _total_tail_carbons(mol: Mol) -> int:
    """Approximate total sp3/sp2 carbon count outside rings — proxy for tail length."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and not atom.IsInRing():
            count += 1
    return count


def _count_ester_per_chain(mol: Mol) -> float:
    """Average ester bonds per acyl chain (proxy for biodegradability)."""
    ester_pat = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][#6]")
    n_ester = len(mol.GetSubstructMatches(ester_pat))
    ionizable_n = Chem.MolFromSmarts("[NX3;!$(NC=O);!$(NS=O);!$(N=*)]")
    n_chains = max(len(mol.GetSubstructMatches(ionizable_n)), 1)
    return n_ester / n_chains


def _head_tail_mw_ratio(mol: Mol) -> float:
    """Ratio of head-group MW to total MW (crude cone-angle proxy).

    Approximation: atoms within 3 bonds of ionizable nitrogen are 'head'.
    """
    ionizable_n = Chem.MolFromSmarts("[NX3;!$(NC=O);!$(NS=O);!$(N=*)]")
    matches = mol.GetSubstructMatches(ionizable_n)
    if not matches:
        return 0.0

    head_atoms: set[int] = set()
    for match in matches:
        for n_idx in match:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, 3, n_idx)
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                head_atoms.add(bond.GetBeginAtomIdx())
                head_atoms.add(bond.GetEndAtomIdx())
            head_atoms.add(n_idx)

    total_mw = Descriptors.MolWt(mol)
    if total_mw == 0:
        return 0.0

    head_mw = sum(mol.GetAtomWithIdx(i).GetMass() for i in head_atoms)
    return head_mw / total_mw


def _longest_carbon_chain(mol: Mol) -> int:
    """Length of the longest aliphatic carbon chain (proxy for max tail length)."""
    carbon_idxs = {
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
    }

    if not carbon_idxs:
        return 0

    adj: dict[int, list[int]] = {idx: [] for idx in carbon_idxs}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in carbon_idxs and b in carbon_idxs:
            adj[a].append(b)
            adj[b].append(a)

    max_len = 0
    for start in carbon_idxs:
        stack = [(start, {start}, 1)]
        while stack:
            node, visited, length = stack.pop()
            max_len = max(max_len, length)
            for nbr in adj[node]:
                if nbr not in visited:
                    stack.append((nbr, visited | {nbr}, length + 1))
    return max_len


def _n_rotatable_in_tails(mol: Mol) -> int:
    """Rotatable bonds outside ring systems — relates to tail flexibility."""
    count = 0
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondTypeAsDouble() != 1.0:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetDegree() < 2 or b.GetDegree() < 2:
            continue
        count += 1
    return count


class CustomLNPBackend(DescriptorBackend):
    """Domain-specific features for ionizable lipid SAR.

    Two categories:
      1. SMARTS-based motif counts (configurable)
      2. Handcrafted physicochemical features derived from LNP design literature
    """

    def __init__(
        self,
        smarts_motifs: dict[str, str] | None = None,
        compute_handcrafted: bool = True,
    ) -> None:
        from ..config import DEFAULT_SMARTS_MOTIFS

        raw = smarts_motifs if smarts_motifs is not None else dict(DEFAULT_SMARTS_MOTIFS)
        self._smarts: dict[str, Chem.Mol | None] = {}
        for feat_name, smarts_str in raw.items():
            pat = Chem.MolFromSmarts(smarts_str)
            if pat is None:
                logger.warning("Invalid SMARTS pattern for %s: %s", feat_name, smarts_str)
            self._smarts[feat_name] = pat

        self._compute_handcrafted = compute_handcrafted

    @property
    def name(self) -> str:
        return "lnp"

    def compute(self, mols: dict[str, Mol]) -> pd.DataFrame:
        records: dict[str, dict[str, float]] = {}

        for mol_name, mol in mols.items():
            row: dict[str, float] = {}

            # SMARTS motif features
            for feat_name, pattern in self._smarts.items():
                if pattern is None:
                    row[f"smarts_{feat_name}"] = 0
                    continue
                if feat_name.startswith("has_"):
                    row[f"smarts_{feat_name}"] = _has_smarts(mol, pattern)
                else:
                    row[f"smarts_{feat_name}"] = _count_smarts(mol, pattern)

            # Handcrafted LNP features
            if self._compute_handcrafted:
                try:
                    row["lnp_total_tail_carbons"] = _total_tail_carbons(mol)
                    row["lnp_ester_per_chain"] = _count_ester_per_chain(mol)
                    row["lnp_head_tail_mw_ratio"] = _head_tail_mw_ratio(mol)
                    row["lnp_longest_chain"] = _longest_carbon_chain(mol)
                    row["lnp_rotatable_in_tails"] = _n_rotatable_in_tails(mol)
                    row["lnp_fraction_csp3"] = Descriptors.FractionCSP3(mol)
                    row["lnp_num_stereocenters"] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                    row["lnp_heavy_atom_count"] = mol.GetNumHeavyAtoms()
                except Exception:
                    logger.warning("Handcrafted feature failed for %s", mol_name, exc_info=True)

            records[mol_name] = row

        df = pd.DataFrame.from_dict(records, orient="index")
        df.index.name = "name"

        # Prefix all columns
        df.columns = [f"lnp_{c}" if not c.startswith(("smarts_", "lnp_")) else c for c in df.columns]

        logger.info("Custom LNP: computed %d features for %d molecules", df.shape[1], len(records))
        return df
