"""
BaseModel — Abstract base class for all predictive models in the benchmark.

Every model (Tier 1 ML, Tier 2 pretrained, Tier 3 domain SOTA) follows the
same API contract:
    model.fit(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)
    model.save(path)
    model.load(path)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ModelResult:
    """Container for a single model's evaluation results."""
    model_name: str
    split_name: str
    seed: int
    metrics: Dict[str, float] = field(default_factory=dict)
    y_pred: Optional[np.ndarray] = None
    y_true: Optional[np.ndarray] = None
    y_val_pred: Optional[np.ndarray] = None
    y_val_true: Optional[np.ndarray] = None


class BaseModel(ABC):
    """Abstract model with standardised fit/predict/evaluate API."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for input features."""
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute default metric suite on (X, y).

        Override in subclasses to add model-specific metrics.
        """
        y_pred = self.predict(X)
        return compute_metrics(y, y_pred)

    def save(self, path: str) -> None:
        """Save model checkpoint to *path*."""
        import joblib, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    def load(self, path: str) -> "BaseModel":
        """Load model checkpoint from *path*."""
        import joblib
        return joblib.load(path)


# ── Metric suite ────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Standard regression metrics for lipid transfection prediction."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr, pearsonr

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    spr, _ = spearmanr(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Spearman": float(spr),
        "Pearson": float(pcc),
    }