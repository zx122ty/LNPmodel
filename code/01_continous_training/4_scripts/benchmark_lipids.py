"""Compare baseline Uni-Mol2 (84M) vs continued-pretrained checkpoint on ionizable
lipid benchmark datasets.

Uses UniMolRepr to extract [CLS] representations, then 5-fold CV with Ridge
and Random Forest. This avoids the MolTrain path which requires UniMol v1
weights (mol_pre_all_h_220816.pt) that are not available offline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIMOL_TOOLS = str(REPO_ROOT / "4_scripts/Uni-Mol/unimol_tools")
sys.path.insert(0, UNIMOL_TOOLS)

from unimol_tools import UniMolRepr

DATASETS = {
    "Sheet2": {
        "path": str(REPO_ROOT / "1_data/raw/benchmark/Sheet2.csv"),
        "smiles_col": "SMILES",
        "target_col": "RLU (log2)",
        "name": "lumi-lab (n=1920)",
    },
    "Sheet3": {
        # Points at the benchmark workbook, NOT 1_data/raw/benchmark/Sheet3.csv:
        # that CSV was snapshotted before the 2026-09-13 structure fix and has
        # 141 wrong structures (9 of them duplicated). The workbook is the
        # corrected source.
        "path": str(BENCHMARK_XLSX),
        "smiles_col": "SMILES",
        # renamed with an IGROV1_ prefix in the 2026-09-13 structure fix
        "target_col": ("IGROV1_transfection_efficiency", "transfection_efficiency"),
        "name": "Ours (n=768)",
    },
}

BASE_CKPT = str(REPO_ROOT / "2_models/pretrained/checkpoint_84M.pt")
ADAPTED_CKPT = str(
    REPO_ROOT / "2_models/continued_pretrain/unimol2_lipids_84m_v1/checkpoints"
    / "checkpoint_last.pt"
)

N_SPLITS = 5
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark lipid checkpoint via UniMolRepr.")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--output", type=Path, default=REPO_ROOT / "5_outputs/benchmark_results.json")
    return p.parse_args()


def load_data(ds_cfg: dict) -> tuple[np.ndarray, list[str]]:
    p = ds_cfg["path"]
    df = pd.read_excel(p, sheet_name="Sheet3") if p.endswith(".xlsx") else pd.read_csv(p)
    col = ds_cfg["target_col"]
    if isinstance(col, tuple):
        col = next((c for c in col if c in df.columns), None)
        if col is None:
            raise SystemExit(f"target column not found in {list(df.columns)}")
    smi = df[ds_cfg["smiles_col"]].dropna().astype(str).tolist()
    y = df[col].dropna().values
    return y, smi


def get_repr(smiles_list, checkpoint_path, label):
    """Extract [CLS] representations via UniMolRepr, optionally swapping weights."""
    clf = UniMolRepr(
        data_type="molecule",
        model_name="unimolv2",
        model_size="84m",
    )
    # UniMolRepr.__init__ does NOT accept pretrained_model_path — it hardcodes
    # weights from its own directory. Swap the state_dict manually if needed.
    if checkpoint_path is not None:
        import torch
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        clf.model.load_state_dict(sd, strict=False)
        print(f"  [{label}] loaded {len(sd)} keys from {checkpoint_path.name}")
    clf.model.eval()
    result = clf.get_repr(smiles_list, return_atomic_reprs=False)
    out = np.array(result["cls_repr"])
    print(f"  [{label}] {len(smiles_list)} molecules -> {out.shape}")
    return out


def cross_validate(X, y, model_name="ridge"):
    """5-fold CV, return (r2, rmse, spearman) averaged across folds."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    r2s, rmses, sps = [], [], []
    for train_idx, test_idx in kf.split(X):
        if model_name == "ridge":
            m = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        else:
            m = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=4)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        r2s.append(r2_score(y[test_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        sps.append(spearmanr(y[test_idx], pred).statistic)
    return {
        "r2": round(float(np.mean(r2s)), 4),
        "r2_std": round(float(np.std(r2s)), 4),
        "rmse": round(float(np.mean(rmses)), 4),
        "spearman": round(float(np.mean(sps)), 4),
        "spearman_std": round(float(np.std(sps)), 4),
    }


def main():
    args = parse_args()
    print(f"Baseline (pretrained): {BASE_CKPT}")
    print(f"Adapted (LipidFM):    {ADAPTED_CKPT}")
    all_results = {}

    for ds_key in args.datasets:
        cfg = DATASETS[ds_key]
        ds_name = cfg["name"]
        print(f"\n{'='*60}\n  {ds_name}\n{'='*60}")

        y, smiles = load_data(cfg)
        print(f"  Targets: n={len(y)}  mean={y.mean():.4f}  std={y.std():.4f}")

        # Extract representations once per checkpoint
        X_base = get_repr(smiles, Path(BASE_CKPT), "baseline (84M)")
        X_adapt = get_repr(smiles, Path(ADAPTED_CKPT), "adapted (LipidFM)")

        results = {}
        for label, X in [("baseline (84M)", X_base), ("adapted (LipidFM)", X_adapt)]:
            ridge = cross_validate(X, y, "ridge")
            rf = cross_validate(X, y, "rf")
            short = "base" if "baseline" in label else "adapt"
            print(f"  [{short}] Ridge: R²={ridge['r2']:.4f}±{ridge['r2_std']:.4f}  "
                  f"Spearman={ridge['spearman']:.4f}±{ridge['spearman_std']:.4f}")
            print(f"  [{short}] RF:    R²={rf['r2']:.4f}±{rf['r2_std']:.4f}  "
                  f"Spearman={rf['spearman']:.4f}±{rf['spearman_std']:.4f}")
            results[label] = {"ridge": ridge, "rf": rf}

        all_results[ds_key] = results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults -> {args.output}")


if __name__ == "__main__":
    main()