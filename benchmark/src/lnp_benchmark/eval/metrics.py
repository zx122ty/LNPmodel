"""
Evaluation suite: metrics, significance testing, data efficiency, and
interpretability utilities.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── Significance Testing ────────────────────────────────────────────────────

def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """Paired two-tailed t-test between two sets of scores (5 seeds)."""
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
    }


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """Wilcoxon signed-rank test (non-parametric alternative)."""
    stat, p_val = stats.wilcoxon(scores_a, scores_b)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
    }


def aggregate_seeds(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results from multiple seeds.

    Input: list of dicts, each containing 'model_name', 'split_name',
           'seed', and metric keys (R2, Spearman, RMSE, ...).

    The split_name is cleaned to strip the _seedNNN suffix so that
    results across seeds are aggregated into a single row per model per
    split method.

    Returns: DataFrame with mean ± std per (model_name, split_name).
    """
    import re
    df = pd.DataFrame(results)
    # Strip _seedNNN suffix from split_name so random_seed42 → random
    df["split_name"] = df["split_name"].apply(
        lambda x: re.sub(r"_seed\d+$", "", x)
    )
    metric_cols = [c for c in df.columns if c.upper() not in
                   {"MODEL_NAME", "SPLIT_NAME", "SEED", "MODEL", "Y_PRED", "Y_TRUE"}]

    grouped = df.groupby(["model_name", "split_name"])[metric_cols].agg(["mean", "std"])
    # Flatten MultiIndex columns
    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    grouped = grouped.reset_index()
    return grouped


# ── Data Efficiency (Ablation) ──────────────────────────────────────────────

def sample_by_fraction(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly sample *fraction* of the data without replacement."""
    rng = np.random.RandomState(seed)
    n = len(X)
    k = max(1, int(n * fraction))
    idx = rng.choice(n, k, replace=False)
    return X[idx], y[idx]


# ── Error Analysis ──────────────────────────────────────────────────────────

def top_error_molecules(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    smiles_list: List[str],
    top_k: int = 50,
    metric: str = "abs_error",
) -> pd.DataFrame:
    """Return the *top_k* molecules with largest prediction errors.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and predicted values.
    smiles_list : List[str]
        Corresponding SMILES strings.
    top_k : int
        Number of worst predictions to return.
    metric : str
        'abs_error' (default) or 'squared_error'.

    Returns
    -------
    DataFrame with columns: SMILES, y_true, y_pred, error.
    """
    if metric == "abs_error":
        errors = np.abs(y_true - y_pred)
    else:
        errors = (y_true - y_pred) ** 2

    top_idx = np.argsort(errors)[-top_k:][::-1]
    return pd.DataFrame({
        "SMILES": [smiles_list[i] for i in top_idx],
        "y_true": y_true[top_idx],
        "y_pred": y_pred[top_idx],
        "error": errors[top_idx],
    })


# ── Interpretability ────────────────────────────────────────────────────────

def compute_shap_summary(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 100,
) -> Optional[np.ndarray]:
    """Compute SHAP values for a trained model.

    Returns SHAP values array of shape (n_test, n_features), or None if
    shap is not installed.
    """
    try:
        import shap
    except ImportError:
        return None

    # Use a subset of training data as background
    if len(X_train) > n_samples:
        rng = np.random.RandomState(42)
        bg = X_train[rng.choice(len(X_train), n_samples, replace=False)]
    else:
        bg = X_train

    explainer = shap.Explainer(model.predict, bg)
    shap_values = explainer(X_test[:n_samples])
    return shap_values.values


def highlight_substructures(
    smiles_list: List[str],
    attribution_scores: np.ndarray,
    atom_smarts_patterns: Optional[List[str]] = None,
) -> List[Dict]:
    """Map high-attribution substructures to SMARTS patterns.

    Returns a list of dicts per molecule with highlighted substructures.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    results = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        results.append({
            "SMILES": smi,
            "num_atoms": mol.GetNumAtoms(),
            # TODO: integrate atom-level attribution -> substructure mapping
        })
    return results