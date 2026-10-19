"""Abstract base class for descriptor backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from rdkit.Chem import Mol


class DescriptorBackend(ABC):
    """Interface that every descriptor backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in column prefixes and logs."""

    @abstractmethod
    def compute(self, mols: dict[str, Mol]) -> pd.DataFrame:
        """Compute descriptors for a set of molecules.

        Args:
            mols: Mapping of molecule name → RDKit Mol object.

        Returns:
            DataFrame indexed by molecule name, one column per descriptor.
            Column names should be prefixed with ``self.name + "_"``.
        """

    def available_descriptors(self) -> list[str]:
        """Return names of all descriptors this backend can compute.

        Default implementation returns an empty list; override when the
        full descriptor catalogue is known ahead of time (e.g. RDKit).
        """
        return []
