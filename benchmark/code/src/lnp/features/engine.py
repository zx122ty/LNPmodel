"""Molecular featurisation: layered descriptors for the new project.

Builds a feature matrix from canonical SMILES using the reused
``descriptor_pipeline`` package:

  * RDKit 2D descriptors (full tier-3 set)
  * Morgan / ECFP and MACCS fingerprints (as sparse-ish bit columns)
  * Custom LNP domain features (SMARTS motifs + handcrafted hydrocarbon
    / head-group attributes) — highly relevant for ionizable lipids

Fingerprint PCA is intentionally NOT applied here at load time; it is applied
*inside* each CV fold (see ``models/pipeline.py``) to prevent leakage. For the
768-sample setting we keep fingerprints as raw bit columns and let the model's
embedded feature selection/pruning handle them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_CODE = Path(__file__).resolve().parents[3]  # benchmark/code/
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402
from lnp.descriptor_pipeline import (  # noqa: E402
    DescriptorCalculator,
    DescriptorPipelineConfig,
    FingerprintConfig,
)
from lnp.descriptor_pipeline.config import FilterConfig  # noqa: E402


def _smiles_to_mols(smiles_list) -> dict[str, Chem.Mol]:
    mols = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        if mol is not None:
            mols[f"mol_{i}"] = mol
    return mols


def _build_calc_config() -> DescriptorPipelineConfig:
    fp_cfg = []
    if config.ENABLE_MORGAN:
        fp_cfg.append(FingerprintConfig(
            fp_type="morgan", radius=config.MORGAN_RADIUS, n_bits=config.MORGAN_NBITS,
            pca_components=None,  # keep raw; PCA handled inside CV
        ))
    if config.ENABLE_MACCS:
        fp_cfg.append(FingerprintConfig(fp_type="maccs", radius=0, n_bits=config.MACCS_NBITS, pca_components=None))

    desc_tier = config.RDKIT_TIER
    return DescriptorPipelineConfig(
        descriptor_tier=desc_tier,
        enable_rdkit=config.ENABLE_RDKIT,
        enable_fingerprints=bool(fp_cfg),
        enable_custom_lnp=config.ENABLE_CUSTOM_LNP,
        enable_mordred=config.ENABLE_MORDRED,
        fingerprints=fp_cfg,
        filter_config=FilterConfig(
            variance_threshold=0.0,
            near_zero_variance_threshold=0.0,
            correlation_threshold=1.0,     # keep full descriptor set for now
            nan_threshold=config._descriptor_nan_threshold if hasattr(config, "_descriptor_nan_threshold") else 0.2,
        ),
    )


def featurize(smiles_list, drop_invalid: bool = True) -> pd.DataFrame:
    """Return a DataFrame of descriptors/fingerprints indexed like ``smiles_list``.

    Raw (deduplicated) output of the descriptor engine, unfiltered except for
    constant column removal. Empty/failed molecules are returned as NaN rows;
    callers should drop them via ``drop_invalid``.
    """
    # Build mol name -> input position map so we can align back to the rows.
    mols = {}
    position = {}
    idx = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        if mol is not None:
            mols[f"mol_{idx}"] = mol
            position[f"mol_{idx}"] = i
            idx += 1

    calc = DescriptorCalculator(_build_calc_config())
    df = calc.compute(mols)  # raw, no filtering (constant cols already dropped)

    # RDKit computes a few descriptors (e.g. Ipc, the information content index)
    # as enormous floats that overflow tree/acc-model float32 buffers. Replace
    # any outlier magnitudes (|x| > 1e6) with NaN so downstream imputation /
    # tree models treat them as missing rather than crashing.
    arr = df.to_numpy(dtype=float)
    outlier = np.abs(arr) > 1e6
    if outlier.any():
        df = df.mask(pd.DataFrame(outlier, index=df.index, columns=df.columns))
        print(f"  [features] masked {int(outlier.sum())} extreme descriptor cells (>1e6) as NaN")

    # Drop columns that are all-NaN after masking (e.g. rdkit_Ipc), so rows are
    # not dropped downstream purely because one uninformative descriptor busted.
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != before:
        print(f"  [features] dropped {before - df.shape[1]} all-NaN columns after masking")

    aligned = pd.DataFrame(index=np.arange(len(smiles_list)), columns=df.columns, dtype=float)
    for name, pos in position.items():
        if name in df.index:
            aligned.iloc[pos] = df.loc[name]
        elif drop_invalid:
            pass  # leave all-NaN row; caller drops
    return aligned