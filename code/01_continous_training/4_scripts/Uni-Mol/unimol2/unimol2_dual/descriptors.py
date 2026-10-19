# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
"""The fixed 32-descriptor MTR target set (plan §5.2 item 4).

Both sides of the MTR objective must agree on names and order:
  * ``build_pretrain_corpus.py`` (repo root, EnviroFM-v2 track) computes raw
    values for every corpus molecule and derives the z-score statistics;
  * :mod:`unimol2_dual.tasks.unimol2_dual` re-computes the same values on the
    fly at training time and z-scores them with the manifest's statistics.

The list is therefore defined exactly once, here, and is never hard-coded in
either consumer. Categories follow plan §5.2 (fundamental / H-bonding /
surface & topology / electronic), completed to 32 with ring and EState
indices. Every descriptor is **conformer-free** (pure 2D) so the target never
depends on which conformer was sampled — only the molecule.

RDKit note: validated against RDKit 2026.03.3, where several legacy names were
removed (``MW`` -> ``MolWt``, ``NumHBondAcceptors`` -> ``NumHAcceptors``,
``Chi*c``/``PET``/``RadiusOfGyration`` gone). Ring counts come from
``rdMolDescriptors.CalcNum{Aliphatic,Aromatic}Rings``.
"""

from __future__ import annotations

import numpy as np

# Order is part of the contract — do not reorder without bumping the manifest.
DESCRIPTOR_NAMES: list[str] = [
    # fundamental (5)
    "MolWt", "MolLogP", "TPSA", "FractionCSP3", "NumRotatableBonds",
    # H-bonding (4)
    "NumHDonors", "NumHAcceptors", "NHOHCount", "NOCount",
    # surface & topology (10)
    "LabuteASA", "BalabanJ", "BertzCT",
    "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
    "HallKierAlpha", "Kappa1",
    # shape / ring (7)
    "Kappa2", "Kappa3", "RingCount", "NumAliphaticRings", "NumAromaticRings",
    "ExactMolWt", "HeavyAtomCount",
    # electronic (2) — Gasteiger partial charges, max/min over heavy atoms
    "MaxPartialCharge", "MinPartialCharge",
    # EState indices (4)
    "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex",
]

assert len(DESCRIPTOR_NAMES) == 32, len(DESCRIPTOR_NAMES)
assert len(set(DESCRIPTOR_NAMES)) == 32


def compute_descriptors(smi: str, names: list[str] | None = None) -> np.ndarray | None:
    """Raw (unz-scored) descriptor vector in the given name order.

    Returns ``None`` when RDKit cannot parse the SMILES; individual failures are
    NaN so a single broken descriptor never drops the molecule (the loss masks
    NaN targets).
    """
    if names is None:
        names = DESCRIPTOR_NAMES
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    charges_cache = None
    vals = []
    for name in names:
        try:
            if name == "NumRotatableBonds":
                v = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
            elif name == "NumAliphaticRings":
                v = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
            elif name == "NumAromaticRings":
                v = float(rdMolDescriptors.CalcNumAromaticRings(mol))
            elif name in ("MaxPartialCharge", "MinPartialCharge"):
                if charges_cache is None:
                    mol_c = Chem.Mol(mol)
                    AllChem.ComputeGasteigerCharges(mol_c)
                    charges_cache = [a.GetDoubleProp("_GasteigerCharge")
                                     for a in mol_c.GetAtoms()
                                     if a.GetSymbol() != "H"]
                v = (float(max(charges_cache)) if name == "MaxPartialCharge"
                     else float(min(charges_cache)))
            else:
                fn = getattr(Descriptors, name, None)
                if fn is None:
                    raise AttributeError(f"unknown descriptor {name!r}")
                v = float(fn(mol))
        except Exception:
            v = float("nan")
        vals.append(v)
    return np.asarray(vals, dtype=np.float32)
