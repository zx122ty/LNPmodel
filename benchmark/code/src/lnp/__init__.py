"""lnp — Ionizable-lipid transfection prediction with uncertainty.

A slim, reproducible package for building a single-target regression model that
predicts log10 transfection efficiency from lipid SMILES and returns calibrated
prediction intervals. Reuses the ``descriptor_pipeline`` package for molecular
descriptors / fingerprints.
"""

from .data.loader import load_dataset, clean_smiles_dataset
from .splits.splitters import RandomSplitter, ScaffoldSplitter, ClusterSplitter

__all__ = [
    "load_dataset",
    "clean_smiles_dataset",
    "RandomSplitter",
    "ScaffoldSplitter",
    "ClusterSplitter",
]