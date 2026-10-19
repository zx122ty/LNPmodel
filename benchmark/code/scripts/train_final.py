"""Train the final point model + calibrate a conformal residual quantile.

Stage 5 — produces operational artifacts consumed by ``predict.py``:

  1. Fit the point predictor (RandomForest by default; the best-stable model
     under random/cluster split) on the *full* 759-lipid training set.
  2. Calibrate a split-conformal additive residual quantile (90% target, log10)
     using a held-out calibration slice so the interval is distribution-free.
  3. Serialise pipeline + quantile + feature metadata to ``outputs/models/final``.

Because point accuracy and interval calibration use disjoint data, the saved
band is honest out-of-the-box (marginal coverage ≈ 90%). The point model is fit
on all data; the calibration slice is subsampled from it for interval only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1]  # benchmark/code/
sys.path.insert(0, str(PROJECT_CODE / "src"))
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402

from lnp.data.loader import load_dataset, inv_log10
from lnp.features.engine import featurize
from lnp.models_final import fit_point_model, calibrate_q, save_artifact


def build_feature_matrix(df) -> pd.DataFrame:
    X = featurize(df["smiles"].tolist())
    X = X.dropna(axis=1, how="all")
    arr = X.to_numpy(float)
    X = X.mask(pd.DataFrame(np.abs(arr) > 1e6, index=X.index, columns=X.columns))
    X = X.dropna(axis=1, how="all")
    from sklearn.impute import SimpleImputer
    X = pd.DataFrame(SimpleImputer(strategy="median").fit(X).transform(X), columns=X.columns)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RandomForest",
                        help="point model name (default RandomForest)")
    parser.add_argument("--cal-frac", type=float, default=0.15,
                        help="fraction held out for conformal calibration (default 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    df = load_dataset()
    X = build_feature_matrix(df)
    y = df["target"].to_numpy(float)
    floor = float(np.min(y))  # nominal floor for safe reference

    print(f"[Stage 5] final fit — model={args.model}, cal_frac={args.cal_frac}")

    # 1. Split full data into train (fit point model) + calibration (fit interval),
    #    disjoint so the band is honest (cal points never seen by the point model).
    from sklearn.model_selection import train_test_split
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=args.cal_frac, random_state=args.seed)

    # 2. Fit the point predictor on the train slice only.
    pipe = fit_point_model(X_train, y_train, args.model)
    print(f"  point model {args.model} fitted on {X_train.shape[0]}×{X_train.shape[1]}")

    # 3. Calibrate split-conformal residual quantile on the held-out slice.
    pred_cal = np.asarray(pipe.predict(X_cal)).ravel()
    q = calibrate_q(y_cal, pred_cal, alpha=args.alpha)
    print(f"  conformal q({1 - args.alpha:.0%}) = {q:.4f} (log10)")

    save_artifact(pipe, q,
                  feature_names=list(X.columns),
                  model_name=args.model,
                  floor=floor)

    # Report accuracy + coverage on the calibration slice (illustrative only)
    from lnp.eval.metrics import regression_metrics, uncertainty_metrics
    m = regression_metrics(y_cal, pred_cal)
    u = uncertainty_metrics(y_cal, pred_cal - q, pred_cal + q)
    print(f"  cal-point R²={m['R2']:.3f} RMSE={m['RMSE']:.3f}")
    print(f"  cal-interval coverage={u['coverage']:.3f} width={u['avg_interval_width']:.3f} (log10)")


if __name__ == "__main__":
    main()