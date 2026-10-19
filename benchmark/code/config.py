"""Global configuration for the ionizable-lipid transfection benchmark.

This is a slimmed-down reconfiguration of the legacy ``Work/code/config.py``,
scoped to a *single* regression target (transfection efficiency, log10) with
uncertainty quantification. All paths are relative to the project root and
derive from ``benchmark/code/``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent          # benchmark/code/
PROJECT_ROOT = CODE_DIR.parent.parent               # LNPmodel/
BENCHMARK_ROOT = CODE_DIR.parent                    # benchmark/

# ── Input data ──────────────────────────────────────────────────────────────
INPUT_DIR = BENCHMARK_ROOT / "input_data"
DATASET_FILE = INPUT_DIR / "dataset.xlsx"
DATASET_SHEET = "Sheet3"

# Column names in the source workbook
COL_ID = "id"
COL_NAME = "name_ionziable_lipid"
COL_SMILES = "SMILES"
COL_TARGET = "transfection_efficiency"

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = CODE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
PROCESSED_DIR = OUTPUT_DIR / "processed"
FP_FILE = PROCESSED_DIR / "molecular_fingerprints.npz"
DESCRIPTORS_FILE = PROCESSED_DIR / "molecular_descriptors.csv"
MERGED_FILE = PROCESSED_DIR / "clean_dataset.csv"

for _d in (FIG_DIR, TABLE_DIR, MODEL_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
SEEDS = [42, 123, 456, 789, 2026]   # multi-seed runs
N_JOBS = -1

# ── Target handling ─────────────────────────────────────────────────────────
LOG_TARGET = True                    # model log10(efficiency)
LOG_BASE = 10.0
TARGET_UNIT = "RLU (luminescence_normalized)"

# ── Data cleaning ───────────────────────────────────────────────────────────
DROP_DUPLICATE_SMILES = True        # keep first occurrence of identical can. SMILES

# ── Descriptors (reused descriptor_pipeline) ────────────────────────────────
RDKIT_TIER = 3                       # 3 = full RDKit 2D descriptor set
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
MACCS_NBITS = 167
FP_PCA_COMPONENTS = None             # None = keep raw fingerprints

# Feature families to compute
ENABLE_RDKIT = True
ENABLE_MORGAN = True
ENABLE_MACCS = True
ENABLE_CUSTOM_LNP = True             # smarts motifs + handcrafted LNP features
ENABLE_MORDRED = False               # slow / tertiary; off by default in this project

# ── Splits ──────────────────────────────────────────────────────────────────
VAL_RATIO = 0.1
TEST_RATIO = 0.1
CLUSTER_CUTOFF = 0.7                 # Tanimoto threshold for ClusterSplit

# ── Modeling ────────────────────────────────────────────────────────────────
BASE_MODELS = [
    "ElasticNet", "RandomForest", "XGBoost", "SVR", "GPR",
]
ENSEMBLE_NAMES = ["Voting", "Stacking"]
ML_ALGOS = ["XGBoost", "RandomForest"]

# Uncertainty
EFFECTIVE_GET_STD = True

# ── Evaluation ──────────────────────────────────────────────────────────────
FIG_DPI = 300
FIG_FORMAT = "png"