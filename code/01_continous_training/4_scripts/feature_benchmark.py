"""Frozen-feature benchmark for LipidFM v2 on the two transfection datasets.

Modeled on my_code/02_train_models.py, but:
  * loads the *continued-pretrained* LipidFM v2 checkpoint into a fresh
    UniMolV2Model via UniMolRepr (freezes the backbone, extracts [CLS] reprs),
  * runs standard ML regressors on those frozen features,
  * reports R2 / RMSE / Spearman via 5-fold CV.

This is a *frozen-feature* evaluation (same口径 as benchmark_lipids.py /
linear probe): the backbone is NOT fine-tuned here, so it only measures the
quality of the learned representations, not end-to-end adaptation. Use this to
smoke-test that the checkpoint loads and produces sane features before running
the heavier end-to-end MolTrain finetune.

Usage:
  conda run -n unimol2 python feature_benchmark.py [--dataset all|Sheet2|Sheet3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]   # -> 01_continous_training/
# Point at OUR vendored unimol_tools so we always use a known-good copy.
_OUR_TOOLS = REPO_ROOT / ".." / "unimol_tools-main"
sys.path.insert(0, str(_OUR_TOOLS))
# Sibling fallback to the symlinked production copy.
sys.path.insert(0, str(REPO_ROOT / "4_scripts/Uni-Mol/unimol_tools"))

import unimol_tools.models.unimolv2 as v2mod
from unimol_tools import UniMolRepr

BENCHMARK_XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
CHECKPOINT = (
    REPO_ROOT / "2_models/continued_pretrain/unimol2_lipids_84m_v2/checkpoints/checkpoint_last.pt"
)
OUTPUT_DIR = REPO_ROOT / "5_outputs/finetune_moltrain"

DATASETS = {
    "Sheet2": {"sheet_name": "Sheet2", "smiles_col": "SMILES", "target_col": "RLU (log2)", "log10": False},
    "Sheet3": {"sheet_name": "Sheet3", "smiles_col": "SMILES",
               "target_col": ("IGROV1_transfection_efficiency", "transfection_efficiency"),
               "log10": True},
}


def resolve_col(df, names) -> str:
    """Sheet3's target gained an ``IGROV1_`` prefix in the 2026-09-13 data fix."""
    if isinstance(names, str):
        names = (names,)
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"none of {list(names)} found in {list(df.columns)}")

CV_SEED = 42
N_SPLITS = 5


def load_data(ds_key: str) -> tuple[list[str], np.ndarray]:
    """Return (smiles, target) with Sheet3 pre-log10'd."""
    cfg = DATASETS[ds_key]
    df = pd.read_excel(BENCHMARK_XLSX, sheet_name=cfg["sheet_name"])
    smiles = df[cfg["smiles_col"]].dropna().astype(str).tolist()
    y = df.loc[: len(smiles) - 1, resolve_col(df, cfg["target_col"])].astype(float).values
    if cfg["log10"]:
        y = np.log10(y)
    return smiles, y


def build_model() -> UniMolRepr:
    """Build UniMolRepr and load the LipidFM v2 checkpoint into it."""
    # Neutralize auto-download so the model only has a fresh random head,
    # then swap in our continued-pretrained checkpoint.
    v2mod.UniMolV2Model.load_pretrained_weights = _load_lipidfm

    clf = UniMolRepr(
        data_type="molecule",
        model_name="unimolv2",
        model_size="84m",
        remove_hs=False,
        use_cuda=torch.cuda.is_available(),
    )
    return clf


def _load_lipidfm(self, path=None, strict=False):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"LipidFM checkpoint not found: {CHECKPOINT}")
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    missing, unexpected = self.load_state_dict(sd, strict=False)
    backbone_missing = [k for k in missing if not k.startswith("classification_head")]
    if backbone_missing:
        raise RuntimeError(f"Backbone incomplete: {len(backbone_missing)} keys missing")
    print(f"  [LipidFM] loaded {len(sd)} backbone keys "
          f"(head skipped: {len(missing) - len(backbone_missing)}, unexpected: {len(unexpected)})")


def extract_features(smiles: list[str], model: UniMolRepr) -> np.ndarray:
    result = model.get_repr(smiles, return_atomic_reprs=False)
    return np.array(result["cls_repr"], dtype=np.float32)


def cross_validate(X: np.ndarray, y: np.ndarray, model_name: str = "ridge") -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    r2s, rmses, sps = [], [], []
    for tr, te in kf.split(X):
        if model_name == "ridge":
            m = Ridge(alpha=1.0, random_state=CV_SEED)
        else:
            m = RandomForestRegressor(n_estimators=200, random_state=CV_SEED, n_jobs=4)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], pred))))
        sps.append(spearmanr(y[te], pred).statistic)
    return {
        "r2": round(float(np.mean(r2s)), 4),
        "r2_std": round(float(np.std(r2s)), 4),
        "rmse": round(float(np.mean(rmses)), 4),
        "spearman": round(float(np.mean(sps)), 4),
        "spearman_std": round(float(np.std(sps)), 4),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen-feature benchmark for LipidFM v2.")
    p.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
    print(f"Checkpoint: {CHECKPOINT}")
    model = build_model()
    model.model.eval()

    ds_keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = {}
    for ds_key in ds_keys:
        print(f"\n{'='*60}\n  {ds_key}\n{'='*60}")
        smiles, y = load_data(ds_key)
        print(f"  n={len(smiles)}  target std={y.std():.4f}")
        X = extract_features(smiles, model)
        print(f"  feature shape: {X.shape}")
        ridge = cross_validate(X, y, "ridge")
        rf = cross_validate(X, y, "rf")
        print(f"  Ridge: R²={ridge['r2']:.4f}±{ridge['r2_std']:.4f} "
              f"Spearman={ridge['spearman']:.4f}±{ridge['spearman_std']:.4f}")
        print(f"  RF:    R²={rf['r2']:.4f}±{rf['r2_std']:.4f} "
              f"Spearman={rf['spearman']:.4f}±{rf['spearman_std']:.4f}")
        results[ds_key] = {"ridge": ridge, "random_forest": rf, "n": len(smiles),
                           "target_scale": ("log2(RLU)" if not DATASETS[ds_key]["log10"] else "log10(RLU)")}

    out_path = OUTPUT_DIR / f"feature_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": str(CHECKPOINT),
        "model": "unimolv2-84m (LipidFM continued-pretrained)",
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
