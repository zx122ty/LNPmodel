"""predict.py — CLI to score new lipid SMILES with the trained model + interval.

Usage (from benchmark/code/):
    python scripts/predict.py                          # all rows of the default workbook
    python scripts/predict.py --csv new_lipids.csv --smiles-col SMILES
    python scripts/predict.py --smiles "CCC...=O"      # single SMILES
    python scripts/predict.py --xlsx my.xlsx --sheet Data

Reads new SMILES, computes the same feature matrix used at training, aligns it
to the stored feature columns, and writes:
    log10_pred, lower/upper (log10), back-transformed pred/lower/upper on the
    raw transfection scale, and the interval half-width (log10).

Artifacts read from ``outputs/models/final/`` (produced by train_final.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1]  # benchmark/code/
sys.path.insert(0, str(PROJECT_CODE / "src"))
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402

from lnp.models_final import MODEL_DIR, META_FILE, QFILE, MODEL_FILE, predict_with_interval
from lnp.data.loader import inv_log10


def _align_to_meta(X: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Align X to stored feature_names (add missing as 0.0, drop extras)."""
    cols = meta["feature_names"]
    aligned = pd.DataFrame(0.0, index=X.index, columns=cols)
    for c in cols:
        if c in X.columns:
            aligned[c] = X[c].values
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Score lipid SMILES with trained model")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--csv", help="CSV with a SMILES column")
    src.add_argument("--smiles", help="single SMILES string")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--output", default=None, help="out CSV path")
    parser.add_argument("--no-backtransform", action="store_true",
                        help="keep raw log10 predictions (no 10^x back-transform)")
    args = parser.parse_args()

    # ── gather SMILES ──
    if args.csv:
        src_df = pd.read_csv(args.csv)
        if args.smiles_col not in src_df.columns:
            raise KeyError(f"column '{args.smiles_col}' missing in {args.csv}")
        smiles = src_df[args.smiles_col].fillna("").astype(str).tolist()
        names = src_df.get("name", src_df.index.astype(str)).tolist()
    elif args.smiles:
        smiles = [args.smiles]
        names = ["__input__"]
    else:
        # default: score the benchmark workbook
        from lnp.data.loader import load_dataset
        df = load_dataset()
        smiles = df["smiles"].tolist()
        names = df["name"].tolist()

    X = _load_smiles_features(smiles)
    meta = json.loads((MODEL_DIR / META_FILE).read_text())
    X = _align_to_meta(X, meta)
    y_log, lo_log, hi_log, meta2 = predict_with_interval(X)

    out = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "log10_pred": y_log,
        "log10_lower": lo_log,
        "log10_upper": hi_log,
    })
    if not args.no_backtransform:
        out["pred_raw"] = np.power(10.0, y_log)
        out["lower_raw"] = np.power(10.0, lo_log)
        out["upper_raw"] = np.power(10.0, hi_log)
    out["interval_halfwidth_log10"] = (hi_log - lo_log) / 2.0

    out_path = args.output or config.TABLE_DIR / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  Predictions ({len(out)} rows) → {out_path}")
    print(out.head(10).round(4).to_string(index=False))


def _load_smiles_features(smiles_):
    from lnp.features.engine import featurize
    from sklearn.impute import SimpleImputer

    X = featurize(smiles_)
    X = X.dropna(axis=1, how="all")
    arr = X.to_numpy(float)
    X = X.mask(pd.DataFrame(np.abs(arr) > 1e6, index=X.index, columns=X.columns))
    X = X.dropna(axis=1, how="all")
    X = pd.DataFrame(SimpleImputer(strategy="median").fit(X).transform(X), columns=X.columns)
    return X


if __name__ == "__main__":
    main()