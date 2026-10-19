"""
Tier 1: Classical ML models (XGBoost, LightGBM, Random Forest, etc.)

Each model is a wrapper around the sklearn API, implementing BaseModel.
Supports Optuna hyperparameter optimisation.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import RegressorMixin

from .base import BaseModel, compute_metrics


class SklearnModel(BaseModel):
    """Wrapper for any sklearn-compatible regressor."""

    def __init__(
        self,
        name: str,
        estimator: RegressorMixin,
        seed: int = 42,
    ):
        super().__init__(name, seed)
        self.estimator = estimator

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SklearnModel":
        self.estimator.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


# ── Factory functions ───────────────────────────────────────────────────────

def make_xgboost(seed: int = 42, **kwargs) -> SklearnModel:
    from xgboost import XGBRegressor
    params = dict(
        n_estimators=200, max_depth=10, learning_rate=0.05,
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    params.update(kwargs)
    return SklearnModel("XGBoost", XGBRegressor(**params), seed=seed)


def make_lightgbm(seed: int = 42, **kwargs) -> SklearnModel:
    from lightgbm import LGBMRegressor
    params = dict(
        n_estimators=200, max_depth=10, learning_rate=0.05,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    params.update(kwargs)
    return SklearnModel("LightGBM", LGBMRegressor(**params), seed=seed)


def make_random_forest(seed: int = 42, **kwargs) -> SklearnModel:
    from sklearn.ensemble import RandomForestRegressor
    params = dict(
        n_estimators=200, max_depth=20, random_state=seed, n_jobs=-1,
    )
    params.update(kwargs)
    return SklearnModel("RandomForest", RandomForestRegressor(**params), seed=seed)


def make_mlp(seed: int = 42, **kwargs) -> SklearnModel:
    from sklearn.neural_network import MLPRegressor
    params = dict(
        hidden_layer_sizes=(128, 64, 32),
        learning_rate_init=0.001, max_iter=1000,
        random_state=seed, early_stopping=True,
    )
    params.update(kwargs)
    return SklearnModel("MLP", MLPRegressor(**params), seed=seed)


def make_linear(seed: int = 42, **kwargs) -> SklearnModel:
    from sklearn.linear_model import Ridge
    params = dict(alpha=1.0, random_state=seed)
    params.update(kwargs)
    return SklearnModel("Ridge", Ridge(**params), seed=seed)


def make_svr(seed: int = 42, **kwargs) -> SklearnModel:
    from sklearn.svm import SVR
    params = dict(kernel="rbf", C=10, gamma="scale")
    params.update(kwargs)
    return SklearnModel("SVR", SVR(**params), seed=seed)


# ── Optuna hyperparameter search ────────────────────────────────────────────

def optuna_optimise_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Optuna HPO for XGBoost and return the best hyperparameters."""
    import optuna
    from xgboost import XGBRegressor

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        from scipy.stats import spearmanr
        spr, _ = spearmanr(y_val, y_pred)
        return float(spr)  # maximise Spearman

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params