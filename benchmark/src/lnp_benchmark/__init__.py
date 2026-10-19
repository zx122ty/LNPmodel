"""
lnp_benchmark — A modular, reproducible benchmark framework for ionizable
lipid transfection efficiency prediction.

Quick start:
    from lnp_benchmark.cli.run_benchmark import BenchmarkPipeline
    pipeline = BenchmarkPipeline("configs/default.yaml")
    pipeline.run()
"""

from .data.base import BaseLipidDataset, LipidRecord
from .data.loaders import LNPCSVDataset, LNPExcelDataset
from .splits.base import BaseSplitter, RandomSplitter, ScaffoldSplitter, ClusterSplitter
from .features.base import BaseFeaturizer, MorganFPFeaturizer, RDKitDescriptorFeaturizer, UniMolReprFeaturizer
from .models.base import BaseModel, ModelResult, compute_metrics
from .models.tier1_ml import (
    make_xgboost, make_lightgbm, make_random_forest,
    make_mlp, make_linear, make_svr, optuna_optimise_xgboost,
)
from .eval.metrics import aggregate_seeds, paired_ttest, wilcoxon_test
from .utils.seed import seed_everything

__all__ = [
    "BaseLipidDataset", "LipidRecord",
    "LNPCSVDataset", "LNPExcelDataset",
    "BaseSplitter", "RandomSplitter", "ScaffoldSplitter", "ClusterSplitter",
    "BaseFeaturizer", "MorganFPFeaturizer", "RDKitDescriptorFeaturizer", "UniMolReprFeaturizer",
    "BaseModel", "ModelResult", "compute_metrics",
    "make_xgboost", "make_lightgbm", "make_random_forest",
    "make_mlp", "make_linear", "make_svr", "optuna_optimise_xgboost",
    "aggregate_seeds", "paired_ttest", "wilcoxon_test",
    "seed_everything",
]