"""Descriptor pipeline for LNP ionizable lipid SAR modeling.

A modular, configurable molecular descriptor computation library with:
  - Tiered descriptor strategy (curated → full → Mordred)
  - sklearn-compatible FingerprintTransformer (PCA inside CV folds)
  - SMARTS-based + handcrafted LNP domain features
  - Hash-based caching and post-computation filtering
  - CLI and Python API

Quick start::

    from descriptor_pipeline import DescriptorCalculator, DescriptorPipelineConfig

    config = DescriptorPipelineConfig(descriptor_tier=2)
    calc = DescriptorCalculator(config)
    mols = calc.load_mols_from_dir("path/to/mol_files/")
    descriptors = calc.compute_and_filter(mols)
"""

from .backends import (
    CustomLNPBackend,
    DescriptorBackend,
    FingerprintBackend,
    FingerprintTransformer,
    MordredBackend,
    RDKit2DBackend,
)
from .calculator import DescriptorCalculator
from .config import (
    CURATED_MORDRED_SUBSET,
    CURATED_RDKIT_SUBSET,
    DEFAULT_SMARTS_MOTIFS,
    DescriptorPipelineConfig,
    FilterConfig,
    FingerprintConfig,
)
from .filters import DescriptorFilter, compute_correlation_report

__all__ = [
    "DescriptorCalculator",
    "DescriptorPipelineConfig",
    "DescriptorFilter",
    "FingerprintTransformer",
    "FilterConfig",
    "FingerprintConfig",
    "DescriptorBackend",
    "RDKit2DBackend",
    "FingerprintBackend",
    "CustomLNPBackend",
    "MordredBackend",
    "CURATED_RDKIT_SUBSET",
    "CURATED_MORDRED_SUBSET",
    "DEFAULT_SMARTS_MOTIFS",
    "compute_correlation_report",
]
