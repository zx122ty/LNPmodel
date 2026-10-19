"""Exploratory DL contrast: Deep-Ensemble MLP (torch-free) + optional GNN stub.

This is an *optional exploratory* module, explicitly NOT part of the main
deliverable pipeline. It compares:
  1. Deep-Ensemble: K independent MLPRegressor (sklearn) on feature matrix,
     averaged point predictions + ensemble-variance → uncertainty.
  2. Optional GNN (torch_geometric) — instantiated only if the environment
     provides torch; otherwise skipped with a clear message.

Run:  python scripts/deep_ensemble_explore.py
Outputs: figures + a results table under outputs/tables/06_explore/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1]  # benchmark/code
sys.path.insert(0, str(PROJECT_CODE / "src"))
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402


def _prep():
    from lnp.data.loader import load_dataset
    from lnp.features.engine import featurize
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split

    df = load_dataset()
    X = featurize(df["smiles"].tolist()).dropna(axis=1, how="all")
    arr = X.to_numpy(float)
    X = X.mask(pd.DataFrame(np.abs(arr) > 1e6, index=X.index, columns=X.columns)).dropna(axis=1, how="all")
    X = pd.DataFrame(SimpleImputer(strategy="median").fit(X).transform(X), columns=X.columns)
    y = df["target"].to_numpy(float)
    return df, X, y


def deep_ensemble_mlp(X, y, n_members=10, seed=42, hidden=(256, 128, 64)):
    """Train K MLPs with different seeds; return point/pred-std + metrics."""
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    preds = []
    for k in range(n_members):
        m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                         max_iter=600, random_state=seed * 1000 + k,
                         early_stopping=True, n_iter_no_change=20)
        m.fit(Xtr_s, ytr)
        preds.append(m.predict(Xte_s))
    P = np.vstack(preds)  # (members, n_test)
    mu = P.mean(0)
    std = P.std(0)

    from lnp.eval.metrics import regression_metrics, uncertainty_metrics
    lm = regression_metrics(yte, mu)
    # Gaussian-ish interval: mu ± 1.645*std ≈ 90%
    lo, hi = mu - 1.645 * std, mu + 1.645 * std
    um = uncertainty_metrics(yte, lo, hi)
    rec = {"method": "deep_ensemble_mlp", "n_members": n_members}
    rec.update({k: v for k, v in lm.items()})
    rec.update({f"u_{k}": v for k, v in um.items()})
    return rec, yte, mu, std


def optional_gnn(X, y, seed=42):
    """Try a small GNN if torch_geometric is available; else skip gracefully."""
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError:
        return None, "torch_geometric not installed — GNN skipped (keep lightweight)"
    # NOTE: a real GNN needs MolGraph featurisation; placeholder returns.
    return NotImplemented, ("torch present but no GNN wiring wired here; "
                            "see notes — kept as optional exploration only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, X, y = _prep()
    print(f"[explore] deep-ensemble MLP, {args.members} members")
    rec, yte, mu, std = deep_ensemble_mlp(X, y, n_members=args.members, seed=args.seed)
    print(f"  MLP-ensemble R²={rec['R2']:.3f} RMSE={rec['RMSE']:.3f} "
          f"Spearman={rec['Spearman_rho']:.3f}")
    print(f"  ensemble-std 90% interval: coverage={rec['u_coverage']:.3f} "
          f"width={rec['u_avg_interval_width']:.3f}")

    # optional GNN
    _, msg = optional_gnn(X, y, args.seed)
    print(f"  GNN: {msg}")

    # Table + figure
    from lnp.eval.visualize import plot_predicted_vs_actual
    tbl_dir = config.TABLE_DIR / "06_explore"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.FIG_DIR / "06_explore"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pd.Series(rec).rename("value").to_csv(tbl_dir / "deep_ensemble_mlp.csv")
    r_raw = np.random.RandomState(0)
    subsample = r_raw.choice(len(yte), size=min(150, len(yte)), replace=False)
    plot_predicted_vs_actual(
        yte[subsample], mu[subsample],
        title=f"Deep-Ensemble MLP ({args.members} members) — test",
        fname="deep_ensemble_mlp", out_dir=fig_dir,
        y_lower=mu[subsample] - 1.645 * std[subsample],
        y_upper=mu[subsample] + 1.645 * std[subsample],
    )
    print(f"  outputs → {tbl_dir}, {fig_dir}")


if __name__ == "__main__":
    main()