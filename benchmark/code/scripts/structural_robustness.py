"""Stage 4 — structural split robustness & leakage check.

Quantifies how model performance degrades as the split becomes more
structure-aware (OOD harder):

  random → cluster (Tanimoto 0.7) → scaffold (Bemis-Murcko)

and verifies the duplicate-SMILES handling actually removed near-duplicate
"cliffs" from train/test boundaries. Outputs a degradation summary + a
similarity-leakage diagnostic (Tanimoto between train and test, per split).
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

import lnp  # noqa: E402


def _prep():
    from lnp.data.loader import load_dataset
    from lnp.features.engine import featurize
    from sklearn.impute import SimpleImputer

    df = load_dataset()
    X = featurize(df["smiles"].tolist()).dropna(axis=1, how="all")
    arr = X.to_numpy(float)
    X = X.mask(pd.DataFrame(np.abs(arr) > 1e6, index=X.index, columns=X.columns)).dropna(axis=1, how="all")
    X = pd.DataFrame(SimpleImputer(strategy="median").fit(X).transform(X), columns=X.columns)
    return df, X


def tanimoto_matrix(smiles_list, n_bits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits) for m in mols if m is not None]
    idx = [i for i, m in enumerate(mols) if m is not None]
    n = len(fps)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            mat[i, j] = mat[j, i] = s
    return mat, idx


def train_test_tanimoto(X_tr_names, X_te_names, n_bits=2048):
    """Max per-test-sample similarity to any training sample (coverage/cliff)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    tr_mols = [Chem.MolFromSmiles(s) for s in X_tr_names if s]
    te_mols = [Chem.MolFromSmiles(s) for s in X_te_names if s]
    tr_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits) for m in tr_mols if m is not None]
    te_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits) for m in te_mols if m is not None]
    if not tr_fps or not te_fps:
        return np.array([])
    maxsim = np.array([
        max(DataStructs.TanimotoSimilarity(t, q) for t in tr_fps) for q in te_fps
    ])
    return maxsim


def main():
    parser = argparse.ArgumentParser(description="Stage 4 — structural robustness")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    args = parser.parse_args()

    from lnp.models.pipeline import make_pipeline, get_default_params
    from lnp.splits.splitters import get_splitter
    from lnp.eval.metrics import regression_metrics

    models = args.models or config.BASE_MODELS
    df, X = _prep()
    y = df["target"].to_numpy(float)
    smiles = df["smiles"].tolist()
    print(f"[Stage 4] structural robustness — n={len(df)}")

    fitters = []
    leakage_rows = []
    for split in ["random", "cluster", "scaffold"]:
        for seed in args.seeds:
            sp = get_splitter(split, seed=seed, val_ratio=config.VAL_RATIO,
                              test_ratio=config.TEST_RATIO, cutoff=config.CLUSTER_CUTOFF)
            s = sp.split(smiles)
            tr, te = s.train, s.test

            # --- leakage diagnostic: max train-tanimoto per test mol ---
            maxsim = train_test_tanimoto([smiles[i] for i in tr], [smiles[i] for i in te])
            leakage_rows.append({
                "split": split, "seed": seed,
                "n_train": len(tr), "n_test": len(te),
                "maxsim_mean": float(maxsim.mean()) if len(maxsim) else np.nan,
                "maxsim_p90": float(np.percentile(maxsim, 90)) if len(maxsim) else np.nan,
                "frac_sim_gt_0.5": float((maxsim > 0.5).mean()) if len(maxsim) else np.nan,
                "frac_sim_gt_0.7": float((maxsim > 0.7).mean()) if len(maxsim) else np.nan,
            })

            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y[tr], y[te]
            for model in models:
                try:
                    params = {"n_features": X_tr.shape[1]}
                    params.update(get_default_params(model))
                    pipe = make_pipeline(model, params)
                    pipe.fit(X_tr, y_tr)
                    pred = np.asarray(pipe.predict(X_te)).ravel()
                    m = regression_metrics(y_te, pred)
                    m["model"] = model
                    m["split"] = split
                    m["seed"] = seed
                    fitters.append(m)
                except Exception as e:
                    print(f"    {split}/s{seed} {model} failed: {e}")

    rec = pd.DataFrame(fitters)
    tbl_dir = config.TABLE_DIR / "04_structural"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # Summary per (split, model): mean over seeds
    agg = rec.groupby(["split", "model"]).agg(
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        Spearman_mean=("Spearman_rho", "mean"), Spearman_std=("Spearman_rho", "std"),
    ).reset_index()
    agg.to_csv(tbl_dir / "structural_summary.csv", index=False)

    lea = pd.DataFrame(leakage_rows)
    lea.to_csv(tbl_dir / "split_leakage_diagnostic.csv", index=False)

    # Figure: per-model metric lines across split hardness
    from lnp.eval.visualize import save_fig
    import matplotlib.pyplot as plt

    order = ["random", "cluster", "scaffold"]
    for metric, ylab, rev in [("R2_mean", "R²", False), ("Spearman_mean", "Spearman ρ", False),
                              ("RMSE_mean", "RMSE", True)]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in rec["model"].unique():
            sub = agg[agg["model"] == model]
            sub = sub.set_index("split").reindex(order)
            ax.plot(order, sub[metric], marker="o", label=model)
        ax.set_xlabel("Split strategy")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} by split (OOD hard-ness)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, f"split_{metric.lower()}", out_dir=config.FIG_DIR / "04_structural")

    print("\n  === Structural summary (R² mean) ===")
    print(agg.pivot(index="model", columns="split", values="R2_mean").round(4).to_string())
    print("\n  === Leakage diagnostic (mean max train-sim per test mol) ===")
    print(lea.pivot_table(index="split", values="maxsim_mean", aggfunc="mean").round(4).to_string())
    print(f"\n  Outputs → {tbl_dir}")
    return agg


if __name__ == "__main__":
    main()