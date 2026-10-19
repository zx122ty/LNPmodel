"""Split-conformal prediction for calibrated transfection intervals.

Implements split-conformal prediction (Papadopoulos et al.) on the residual
`r = |y - ŷ|` of a base point predictor. Calibrated on a held-out calibration
fold (or the val slice) which is disjoint from both training and test.

For a user-chosen alpha (default 0.1 → 90% coverage), the quantile of
calibration residuals gives an additive interval in log10 space:
    [ŷ - q, ŷ + q]

where q = quantile(r_cal, ceil((n+1)(1-alpha))/n). Under exchangeability this
guarantees marginal coverage >= 1-alpha regardless of the base model, making it
distribution-free — exactly suited to the skewed transfection target.
"""

from __future__ import annotations

import numpy as np


def conformal_residual_quantile(
    residuals: np.ndarray,
    alpha: float = 0.05,
    method: str = "conformal",
) -> float:
    """Return the (1-alpha) empirical quantile of |residual|s.

    Conformal uses ceil(((n+1) * (1-alpha)) / n) as the quantile level (finite-
    sample, slightly conservative). ``method='naive'`` uses the plain (1-alpha)
    quantile for comparison.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    if method == "conformal":
        level = np.ceil((n + 1) * (1 - alpha)) / n
        level = np.clip(level, 1e-9, 1.0)
        q = float(np.quantile(residuals, level))
    else:
        q = float(np.quantile(residuals, 1 - alpha))
    return q


def apply_intervals(pred: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Build lower/upper prediction-interval bounds in log10 space."""
    y_pred = np.asarray(pred, dtype=float)
    return y_pred - q, y_pred + q


def calibrate_and_interval(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    alpha: float = 0.05,
    method: str = "conformal",
) -> tuple[np.ndarray, np.ndarray, float]:
    """One-shot split-conformal: fit q on calibration, apply to test preds.

    Returns (lower, upper, q). All in log10 space.
    """
    residuals = np.abs(np.asarray(pred_cal, float) - np.asarray(y_cal, float))
    q = conformal_residual_quantile(residuals, alpha=alpha, method=method)
    lo, hi = apply_intervals(pred_test, q)
    return lo, hi, q


def quantile_forest_intervals(
    rf_lower, rf_upper
) -> tuple[np.ndarray, np.ndarray]:
    """Pass-through helper for precomputed QuantileRF lower/upper bounds."""
    return np.asarray(rf_lower, float), np.asarray(rf_upper, float)