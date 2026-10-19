"""Regression metrics and summarisation for the transfection benchmark.

Mirrors the metric suite in the legacy ``utils/metrics.py`` but scoped to
single-target (log10) regression with an added uncertainty-metric set for
Stage-3 conformal calibration work.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Core regression metrics (all in log10 target space)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    spr = stats.spearmanr(y_true, y_pred).statistic
    pcc = stats.pearsonr(y_true, y_pred).statistic
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Spearman_rho": float(spr),
        "Pearson_r": float(pcc),
    }


def uncertainty_metrics(y_true, y_lower, y_upper, y_pred=None) -> dict:
    """Coverage / interval-width metrics for calibrated prediction intervals.

    Computed in log10 space. ``y_lower``/``y_upper`` are the ccalibrated bounds.
    """
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(y_lower, dtype=float)
    hi = np.asarray(y_upper, dtype=float)
    in_interval = (y_true >= lo) & (y_true <= hi)
    coverage = float(in_interval.mean())
    width = float(np.mean(hi - lo))
    # Normalised (PICP-width) score: lower is better
    ncv = float(np.mean((hi - lo) / np.maximum(np.abs(y_true), 1e-6)))
    return {
        "coverage": coverage,
        "avg_interval_width": width,
        "normalized_interval_width": ncv,
    }


def cv_summary(fold_metrics: list) -> dict:
    """fold_metrics -> dict of 'mean ± std' strings (back-compat with legacy)."""
    keys = fold_metrics[0].keys()
    return {k: f"{np.mean([m[k] for m in fold_metrics]):.4f} ± "
                f"{np.std([m[k] for m in fold_metrics]):.4f}" for k in keys}


def cv_summary_numeric(fold_metrics: list) -> dict:
    """Return per-metric mean/std dicts."""
    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out