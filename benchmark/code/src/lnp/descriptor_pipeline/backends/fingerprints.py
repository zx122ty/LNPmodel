"""Fingerprint computation backend and sklearn-compatible PCA transformer.

The ``FingerprintTransformer`` is designed to be used **inside** cross-validation
folds so that PCA is fit only on training data, preventing information leakage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from .base import DescriptorBackend

if TYPE_CHECKING:
    from rdkit.Chem import Mol

    from ..config import FingerprintConfig

logger = logging.getLogger(__name__)


def _compute_single_fp(mol: Mol, cfg: FingerprintConfig) -> np.ndarray:
    """Compute a single fingerprint for one molecule."""
    if cfg.fp_type == "morgan":
        gen = GetMorganGenerator(radius=cfg.radius, fpSize=cfg.n_bits)
        fp = gen.GetFingerprint(mol)
    elif cfg.fp_type == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif cfg.fp_type == "atom_pair":
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=cfg.n_bits)
    elif cfg.fp_type == "topological":
        fp = RDKFingerprint(mol, fpSize=cfg.n_bits)
    else:
        raise ValueError(f"Unknown fingerprint type: {cfg.fp_type}")
    return np.array(fp, dtype=np.uint8)


class FingerprintBackend(DescriptorBackend):
    """Compute molecular fingerprints as raw bit vectors.

    Returns raw (unreduced) fingerprints. Use ``FingerprintTransformer``
    for PCA reduction inside CV folds.
    """

    def __init__(self, configs: list[FingerprintConfig]) -> None:
        self._configs = configs

    @property
    def name(self) -> str:
        return "fingerprint"

    def compute(self, mols: dict[str, Mol]) -> pd.DataFrame:
        names = sorted(mols.keys())
        frames: list[pd.DataFrame] = []

        for cfg in self._configs:
            n_bits = 167 if cfg.fp_type == "maccs" else cfg.n_bits
            arr = np.zeros((len(names), n_bits), dtype=np.uint8)

            for i, mol_name in enumerate(names):
                try:
                    arr[i] = _compute_single_fp(mols[mol_name], cfg)
                except Exception:
                    logger.warning("Fingerprint %s failed for %s", cfg.name, mol_name, exc_info=True)

            col_names = [f"fp_{cfg.name}_{j}" for j in range(arr.shape[1])]
            df = pd.DataFrame(arr, index=names, columns=col_names)
            df.index.name = "name"

            # Remove constant-bit columns
            var_mask = df.var() > 0
            n_removed = (~var_mask).sum()
            if n_removed > 0:
                df = df.loc[:, var_mask]
                logger.info("FP %s: removed %d constant bits, %d remain", cfg.name, n_removed, var_mask.sum())

            frames.append(df)
            logger.info("FP %s: %d bits for %d molecules", cfg.name, df.shape[1], len(names))

        if not frames:
            return pd.DataFrame(index=pd.Index(names, name="name"))

        return pd.concat(frames, axis=1)

    def compute_raw_arrays(self, mols: dict[str, Mol]) -> dict[str, tuple[np.ndarray, list[str]]]:
        """Return raw fingerprint arrays keyed by config name.

        Returns:
            Dict of {fp_name: (bit_array [n_mols, n_bits], mol_names)}.
        """
        names = sorted(mols.keys())
        result = {}

        for cfg in self._configs:
            n_bits = 167 if cfg.fp_type == "maccs" else cfg.n_bits
            arr = np.zeros((len(names), n_bits), dtype=np.uint8)
            for i, mol_name in enumerate(names):
                try:
                    arr[i] = _compute_single_fp(mols[mol_name], cfg)
                except Exception:
                    logger.warning("Fingerprint %s failed for %s", cfg.name, mol_name, exc_info=True)
            result[cfg.name] = (arr, names)

        return result


class FingerprintTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that reduces fingerprint columns via PCA.

    Designed to be used inside a sklearn Pipeline within CV folds, so PCA
    is fit only on training data (no information leakage).

    Args:
        n_components: Number of PCA components. If None, no reduction is applied.
        fp_prefix: Column prefix to identify fingerprint columns (e.g. "fp_morgan").
        random_state: Random seed for PCA.
    """

    def __init__(
        self,
        n_components: int | None = 30,
        fp_prefix: str = "fp_",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.fp_prefix = fp_prefix
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: object = None) -> FingerprintTransformer:
        if self.n_components is None:
            return self

        fp_cols = [c for c in X.columns if c.startswith(self.fp_prefix)]
        self.fp_cols_ = fp_cols
        self.other_cols_ = [c for c in X.columns if c not in fp_cols]

        if not fp_cols:
            return self

        n_comp = min(self.n_components, len(fp_cols), X.shape[0] - 1)
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca_.fit(X[fp_cols].values.astype(float))
        self.n_components_actual_ = n_comp

        logger.info(
            "FingerprintTransformer: PCA %d → %d components (%.1f%% variance)",
            len(fp_cols), n_comp, self.pca_.explained_variance_ratio_.sum() * 100,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.n_components is None:
            return X

        if not hasattr(self, "fp_cols_") or not self.fp_cols_:
            return X

        fp_pca = self.pca_.transform(X[self.fp_cols_].values.astype(float))
        pc_cols = [f"{self.fp_prefix}PC{i + 1}" for i in range(fp_pca.shape[1])]
        pca_df = pd.DataFrame(fp_pca, index=X.index, columns=pc_cols)

        other_df = X[self.other_cols_]
        return pd.concat([other_df, pca_df], axis=1)

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        if self.n_components is None or not hasattr(self, "fp_cols_") or not self.fp_cols_:
            return list(input_features) if input_features is not None else []
        pc_cols = [f"{self.fp_prefix}PC{i + 1}" for i in range(self.n_components_actual_)]
        return self.other_cols_ + pc_cols
