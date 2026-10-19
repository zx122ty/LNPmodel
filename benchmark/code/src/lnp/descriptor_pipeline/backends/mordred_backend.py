"""Optional Mordred descriptor backend (lazy-imported).

Mordred is not a required dependency. This backend gracefully degrades if
mordred is not installed, raising a clear error only when actually invoked.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import DescriptorBackend

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


def _check_mordred_available() -> bool:
    try:
        import mordred  # noqa: F401
        return True
    except ImportError:
        return False


class MordredBackend(DescriptorBackend):
    """Compute Mordred molecular descriptors.

    Args:
        subset: Compute only these descriptors. If None and use_curated is
            False, computes all ~1800 descriptors (**not recommended** for
            small datasets).
        use_curated: If True and subset is None, use the curated SAR-relevant
            subset (~50 descriptors) instead of the full set.
        ignore_3d: Skip 3D-dependent descriptors (recommended for 2D inputs).
    """

    def __init__(
        self,
        subset: list[str] | None = None,
        use_curated: bool = True,
        ignore_3d: bool = True,
    ) -> None:
        self._subset = subset
        self._use_curated = use_curated
        self._ignore_3d = ignore_3d

    @property
    def name(self) -> str:
        return "mordred"

    def compute(self, mols: dict[str, Mol]) -> pd.DataFrame:
        if not _check_mordred_available():
            raise ImportError(
                "mordred is not installed. Install it with: "
                "pip install mordred  (or pip install mordredcommunity for maintained fork)"
            )

        from mordred import Calculator, descriptors

        calc = Calculator(descriptors, ignore_3D=self._ignore_3d)

        all_desc_names = [str(d) for d in calc.descriptors]
        logger.info("Mordred: %d total descriptors available", len(all_desc_names))

        # Determine which descriptors to keep
        if self._subset is not None:
            keep_set = set(self._subset)
        elif self._use_curated:
            from ..config import CURATED_MORDRED_SUBSET
            keep_set = set(CURATED_MORDRED_SUBSET)
        else:
            keep_set = None
            logger.warning(
                "Computing ALL %d Mordred descriptors. This is not recommended "
                "for datasets with n < 500. Consider using a curated subset.",
                len(all_desc_names),
            )

        names = sorted(mols.keys())
        mol_list = [mols[n] for n in names]

        # Compute all, then filter — Mordred's Calculator doesn't support
        # selective computation efficiently
        logger.info("Mordred: computing descriptors for %d molecules...", len(mol_list))
        df = calc.pandas(mol_list)
        df.index = pd.Index(names, name="name")

        # Mordred returns error objects for failed descriptors — coerce to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter to requested subset
        if keep_set is not None:
            available = [c for c in df.columns if str(c) in keep_set]
            missing = keep_set - set(str(c) for c in df.columns)
            if missing:
                logger.warning("Mordred subset: %d requested descriptors not found: %s",
                               len(missing), sorted(missing)[:10])
            df = df[available]

        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Prefix columns
        df.columns = [f"mordred_{c}" for c in df.columns]

        n_nan_cols = df.isna().any().sum()
        logger.info(
            "Mordred: computed %d descriptors for %d molecules (%d cols have NaN)",
            df.shape[1], len(names), n_nan_cols,
        )
        return df
