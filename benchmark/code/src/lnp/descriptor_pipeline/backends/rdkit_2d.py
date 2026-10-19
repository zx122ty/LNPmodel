"""RDKit 2D molecular descriptor backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from .base import DescriptorBackend

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)

_ALL_RDKIT_NAMES: list[str] = [d[0] for d in Descriptors.descList]


class RDKit2DBackend(DescriptorBackend):
    """Compute RDKit 2D molecular descriptors.

    Args:
        subset: Compute only these descriptors. ``None`` computes all ~210.
    """

    def __init__(self, subset: list[str] | None = None) -> None:
        if subset is not None:
            unknown = set(subset) - set(_ALL_RDKIT_NAMES)
            if unknown:
                logger.warning("Unknown RDKit descriptors (will be skipped): %s", unknown)
            self._desc_names = [n for n in subset if n in set(_ALL_RDKIT_NAMES)]
        else:
            self._desc_names = list(_ALL_RDKIT_NAMES)

        self._calc = MoleculeDescriptors.MolecularDescriptorCalculator(self._desc_names)

    @property
    def name(self) -> str:
        return "rdkit"

    def available_descriptors(self) -> list[str]:
        return list(_ALL_RDKIT_NAMES)

    def compute(self, mols: dict[str, Mol]) -> pd.DataFrame:
        records: dict[str, dict[str, float]] = {}
        failed: list[str] = []

        for mol_name, mol in mols.items():
            try:
                vals = self._calc.CalcDescriptors(mol)
                records[mol_name] = dict(zip(self._desc_names, vals))
            except Exception:
                logger.warning("Descriptor calculation failed for %s", mol_name, exc_info=True)
                failed.append(mol_name)

        if failed:
            logger.warning("Failed to compute descriptors for %d molecules: %s", len(failed), failed)

        df = pd.DataFrame.from_dict(records, orient="index")
        df.index.name = "name"

        # Replace inf with NaN for consistent downstream handling
        df = df.replace([np.inf, -np.inf], np.nan)

        # Prefix columns
        df.columns = [f"rdkit_{c}" for c in df.columns]

        logger.info(
            "RDKit 2D: computed %d descriptors for %d molecules (%d failed)",
            len(self._desc_names), len(records), len(failed),
        )
        return df
