"""Final model fitting + serialisation utilities.

Encapsulates the "training data prep → fit best model → save pipeline + fitted
calibration quantile" workflow so both ``train_final.py`` and ``predict.py``
share one source of truth for the artifacts' layout.

Layout of saved artifacts under ``outputs/models/final/``:
    rf.joblib        fitted RandomForest/XGBoost point predictor
    scaler.joblib    optional standard scaler used at predict time
    meta.json        model name, feature columns, split/cv seed, floor
    conformal_q.json calibrated additive residual quantile (log10)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[3]  # benchmark/code/
sys.path.insert(0, str(PROJECT_CODE / "src"))
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402


MODEL_DIR = config.MODEL_DIR / "final"
MODEL_FILE = "point_model.job"
META_FILE = "meta.json"
QFILE = "conformal_q.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── building the point predictor (fold-internal-stable) ─────────────────────

def make_point_pipeline(model_name: str = "RandomForest", n_features: int = 1000):
    """Return an unfitted sklearn Pipeline for the chosen point model.

    Uses the per-model default params from ``models.pipeline`` (feature scaling
    / selection inside the pipeline, so it's safe to fit on any subset).
    """
    from lnp.models.pipeline import make_pipeline, get_default_params

    params = {"n_features": n_features}
    params.update(get_default_params(model_name))
    return make_pipeline(model_name, params)


def fit_point_model(X: pd.DataFrame, y: np.ndarray, model_name: str = "RandomForest"):
    """Fit a point predictor on the full feature matrix + y."""
    pipe = make_point_pipeline(model_name, n_features=X.shape[1])
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------

def calibrate_q(y_cal: np.ndarray, pred_cal: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal residual quantile on a calibration set (in log10 space)."""
    from lnp.models.conformal import conformal_residual_quantile

    resid = np.abs(np.asarray(pred_cal, float) - np.asarray(y_cal, float))
    return conformal_residual_quantile(resid, alpha=alpha)


def save_artifact(pipeline, q: float, feature_names: list,
                  model_name: str, floor: float):
    """Persist point model + calibrated quantile + metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / MODEL_FILE)
    (MODEL_DIR / QFILE).write_text(json.dumps({"q": float(q), "alpha": 0.1}))
    (MODEL_DIR / META_FILE).write_text(json.dumps({
        "model": model_name,
        "log_base": config.LOG_BASE,
        "floor": float(floor),
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }))
    print(f"  [final] saved → {MODEL_DIR}")


def predict_with_interval(X: pd.DataFrame, alpha: float = 0.1
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load artifacts and return (log10_pred, lower_log10, upper_log10) for X.

    ``aligned_cols`` = the stored ``feature_names``; caller must align X to them
    (missing cols → 0.0) before calling.
    """
    meta = json.loads((MODEL_DIR / META_FILE).read_text())
    q = json.loads((MODEL_DIR / QFILE).read_text())["q"]
    pipe = joblib.load(MODEL_DIR / MODEL_FILE)

    y_pred = np.asarray(pipe.predict(X)).ravel()
    lo = y_pred - q
    hi = y_pred + q
    return y_pred, lo, hi, meta