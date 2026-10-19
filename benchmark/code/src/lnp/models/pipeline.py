"""sklearn pipeline builder for per-model training.

Feature engineering is deliberately generic here: each model receives the full
descriptor/fingerprint matrix, and feature selection / scaling / PCA live
*inside* the sklearn Pipeline so they are fit only on training folds (no CV
leakage). FP columns are recognised by the ``fp_`` prefix and can be PCA-reduced
with a fold-internal ``FingerprintTransformer``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

PROJECT_CODE = Path(__file__).resolve().parents[3]  # benchmark/code/
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402

from lnp.descriptor_pipeline.backends.fingerprints import FingerprintTransformer  # noqa: E402

DEFAULT_PARAMS = {
    "ElasticNet": {"select__k": 50, "model__alpha": 0.1, "model__l1_ratio": 0.5},
    "RandomForest": {"model__n_estimators": 400, "model__max_depth": None,
                     "model__min_samples_leaf": 2, "model__max_features": "sqrt"},
    "XGBoost": {"model__n_estimators": 400, "model__max_depth": 6,
                "model__learning_rate": 0.05, "model__subsample": 0.8,
                "model__colsample_bytree": 0.8, "model__reg_alpha": 0.1,
                "model__reg_lambda": 1.0},
    "SVR": {"select__k": 50, "model__C": 10.0, "model__epsilon": 0.1, "model__gamma": "scale"},
    "GPR": {"fp_pca__n_components": 20, "select__k": 30,
            "model__length_scale": 1.0, "model__noise_level": 0.1},
}


def _gpr_kernel(length_scale, noise_level):
    return ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)


def _clamp_k(k, n_features):
    if isinstance(k, str):
        return k
    return min(int(k), max(1, n_features))


def make_pipeline(name: str, params: dict) -> Pipeline:
    """Build a fresh Pipeline for ``name``, parameterised by flat ``params``.

    Selector/scaler/PCA steps are all fold-internal.
    """
    seed = config.RANDOM_SEED
    n_feat = params.get("n_features", 10_000)
    fp_pca = params.get("fp_pca", True)

    fp_steps = []
    if fp_pca:
        fp_steps.append(("fp_pca", FingerprintTransformer(
            n_components=params.get("fp_pca__n_components"),
            fp_prefix="fp_", random_state=seed)))

    def with_var(*steps):
        return Pipeline(list(fp_steps) + [("var", VarianceThreshold(threshold=0.001))] + list(steps))

    if name == "ElasticNet":
        k = _clamp_k(params["select__k"], n_feat)
        return with_var(
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_regression, k=k)),
            ("model", ElasticNet(alpha=params["model__alpha"], l1_ratio=params["model__l1_ratio"],
                                 max_iter=10000, random_state=seed)),
        )
    if name == "RandomForest":
        return with_var(
            ("model", RandomForestRegressor(
                n_estimators=params["model__n_estimators"],
                max_depth=params["model__max_depth"],
                min_samples_leaf=params["model__min_samples_leaf"],
                max_features=params["model__max_features"],
                random_state=seed, n_jobs=1)),
        )
    if name == "XGBoost":
        return with_var(
            ("model", xgb.XGBRegressor(
                n_estimators=params["model__n_estimators"],
                max_depth=params["model__max_depth"],
                learning_rate=params["model__learning_rate"],
                subsample=params["model__subsample"],
                colsample_bytree=params["model__colsample_bytree"],
                reg_alpha=params["model__reg_alpha"],
                reg_lambda=params["model__reg_lambda"],
                random_state=seed, n_jobs=1, verbosity=0)),
        )
    if name == "SVR":
        k = _clamp_k(params["select__k"], n_feat)
        return with_var(
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_regression, k=k)),
            ("model", SVR(kernel="rbf", C=params["model__C"],
                          epsilon=params["model__epsilon"], gamma=params["model__gamma"])),
        )
    if name == "GPR":
        k = _clamp_k(params["select__k"], n_feat)
        return with_var(
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_regression, k=k)),
            ("model", GaussianProcessRegressor(
                kernel=_gpr_kernel(params["model__length_scale"], params["model__noise_level"]),
                alpha=1e-6, random_state=seed, normalize_y=True)),
        )
    raise ValueError(f"Unknown model name: {name}")


def get_default_params(name: str) -> dict:
    return dict(DEFAULT_PARAMS[name])


AVAILABLE_MODELS = list(DEFAULT_PARAMS.keys())