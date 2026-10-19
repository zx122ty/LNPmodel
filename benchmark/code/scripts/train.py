"""Stage 2 — feature engineering + baseline model training & evaluation.

Flow per (split method × seed):
  1. Load / clean dataset (759 rows after dedup).
  2. Build full feature matrix (RDKit 2D + Morgan/MACCS + LNP custom).
  3. Build train/val/test folds via the chosen splitter.
  4. For each model, run CV scoring on train/val splits, record OOF predictions.
  5. Aggregate metrics per (split × model), save tables + figures.

All feature selection / scaling / FP-PCA happen inside each model's sklearn
Pipeline (fit only on training folds) → no data leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1]  # benchmark/code/
sys.path.insert(0, str(PROJECT_CODE / "src"))
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402

import lnp  # noqa: E402


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full descriptor/fingerprint matrix for the cleaned df."""
    from lnp.features.engine import featurize

    X = featurize(df["smiles"].tolist())
    # Drop all-NaN rows / columns (should be none after cleaning)
    mask = X.notna().all(axis=1)
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"  [features] dropping {dropped} rows with NaN features")
        X = X[mask]
        df = df.loc[X.index].copy()
    return X, df


def run_cv(df, splitter, models):
    """Run nested train/val CV for a single splitter, returning per-model metrics + OOF.

    Returns (records list, oof dict {model: 1D array over the whole dataset}).
    """
    from lnp.splits.splitters import get_splitter as _fac
    from lnp.eval.metrics import regression_metrics
    from lnp.models.pipeline import make_pipeline, get_default_params
    from sklearn.model_selection import KFold

    X, df = get_feature_matrix(df)
    y = df["target"].to_numpy(float)
    names = df["name"].tolist()

    Xc = X.reset_index(drop=True)
    # Impute residual NaN (from masked extreme descriptors). Fit median ONLY on
    # training rows per fold to avoid leakage.
    from sklearn.impute import SimpleImputer
    imp_mean = SimpleImputer(strategy="median")
    records = []
    oof = {m: np.full(len(df), np.nan) for m in models}
    n_seen = {m: np.zeros(len(df), dtype=int) for m in models}

    # Use inner KFold (5) on the splitter's "train" subset per outer fold.
    outer = None
    if splitter.lower().startswith(("random", "r")):
        from sklearn.model_selection import KFold
        outer = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        fold_iter = outer.split(Xc)
    else:
        # structure-aware: single 80/10/10 split by scaffold/cluster
        split_instance = _fac(splitter, seed=config.RANDOM_SEED,
                              val_ratio=config.VAL_RATIO, test_ratio=config.TEST_RATIO,
                              cutoff=config.CLUSTER_CUTOFF)
        s = split_instance.split(df["smiles"].tolist())
        fold_iter = [(s.train, s.test)]  # single held-out test per splitter seed

    for fold_i, (tr_idx, te_idx) in enumerate(fold_iter):
        X_tr, y_tr = Xc.iloc[tr_idx], y[tr_idx]
        X_te, y_te = Xc.iloc[te_idx], y[te_idx]
        # Fit median imputer on training rows only, then transform both.
        imp = SimpleImputer(strategy="median").fit(X_tr)
        X_tr = pd.DataFrame(imp.transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_te = pd.DataFrame(imp.transform(X_te), index=X_te.index, columns=X_te.columns)
        for name in models:
            try:
                params = {"n_features": X_tr.shape[1]}
                params.update(get_default_params(name))
                pipe = make_pipeline(name, params)
                pipe.fit(X_tr, y_tr)
                pred = np.asarray(pipe.predict(X_te)).ravel()
                oof[name][te_idx] = pred
                n_seen[name][te_idx] += 1
                records.append({
                    "split": splitter, "seed": config.RANDOM_SEED, "fold": fold_i,
                    "model": name, **regression_metrics(y_te, pred),
                })
            except Exception as e:
                print(f"    [fold {fold_i}] {name} failed: {e}")
    # average OOF where seen
    oof_avg = {m: np.where(n_seen[m] > 0, oof[m] / np.maximum(n_seen[m], 1), np.nan)
               for m in models}
    return records, oof_avg, y, names


def aggregate(records: list) -> pd.DataFrame:
    """Summarise per (split × model) mean±std across folds."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    grp = df.groupby(["split", "model"])[["R2", "RMSE", "MAE", "Spearman_rho", "Pearson_r"]]
    agg = grp.agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns]
    return agg


def main():
    parser = argparse.ArgumentParser(description="Stage 2 — baseline models")
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold", "cluster"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="override default model set (ElasticNet RandomForest XGBoost SVR GPR)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()

    models = args.models or config.BASE_MODELS
    df = lnp.load_dataset()
    print(f"[Stage 2] baseline — n={len(df)}, models={models}, splits={args.splits}")

    all_rows = []
    oof_store = {}
    for split in args.splits:
        for seed in args.seeds:
            # temporarily set global seed so splitter/config are consistent
            _prev = config.RANDOM_SEED
            config.RANDOM_SEED = seed
            try:
                recs, oof, y, names = run_cv(df, split, models)
                all_rows.extend(recs)
                for m in models:
                    k = f"{split}_s{seed}_{m}"
                    oof_store[k] = {"names": names, "y": y, "pred": oof[m]}
            finally:
                config.RANDOM_SEED = _prev

    rec_df = pd.DataFrame(all_rows).dropna(subset=["R2"])
    tbl_dir = config.TABLE_DIR / "02_baseline"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # Raw per-fold records
    rec_df.to_csv(tbl_dir / "cv_fold_records.csv", index=False)

    # Aggregated summary
    agg = rec_df.groupby(["split", "model"]).agg(
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        Spearman_mean=("Spearman_rho", "mean"), Spearman_std=("Spearman_rho", "std"),
        Pearson_mean=("Pearson_r", "mean"), Pearson_std=("Pearson_r", "std"),
    ).reset_index()
    agg.to_csv(tbl_dir / "baseline_summary.csv", index=False)

    # OOF table (wide) per model for later uncertainty work.
    oof_rows = pd.DataFrame({"name": names, "actual": y})
    for m in models:
        for k, store in oof_store.items():
            if k.endswith(f"_{m}"):
                col = f"pred_{m}"
                oof_rows[col] = oof_rows["name"].map(pd.Series(store["pred"], index=store["names"]))
                break
    oof_rows.to_csv(tbl_dir / "oof_predictions_random.csv", index=False)

    # ── Figures ──
    from lnp.eval.visualize import plot_model_comparison, plot_split_comparison, plot_predicted_vs_actual

    fig_dir = config.FIG_DIR / "02_baseline"
    # Per-split model comparison
    for split in args.splits:
        sub = agg[agg["split"] == split]
        if sub.empty:
            continue
        plot_model_comparison(sub, metric="R2_mean", ylabel="R² mean",
                              title=f"Model R² ({split} split)", fname=f"r2_{split}",
                              out_dir=fig_dir)
    # Split comparison (single-seed scalar view)
    plot_split_comparison(agg, metrics=("R2_mean", "RMSE_mean", "Spearman_mean"),
                          out_dir=fig_dir)

    # Pred-vs-actual for the best random-split model (by mean R²)
    if "random" in args.splits and not oof_rows.empty:
        best_m = rec_df[rec_df["split"] == "random"].groupby("model")["R2"].mean().idxmax()
        if f"pred_{best_m}" in oof_rows.columns:
            va = oof_rows.dropna(subset=["actual", f"pred_{best_m}"])
            plot_predicted_vs_actual(
                va["actual"].values, va[f"pred_{best_m}"].values,
                target_name="log10(transfection)",
                title=f"Random-split OOF — best model: {best_m}",
                fname=f"pred_vs_actual_random_{best_m}", out_dir=fig_dir)

    print("\n  === Baseline summary (R² mean ± std) ===")
    print(agg[["split", "model", "R2_mean", "RMSE_mean", "Spearman_mean"]].round(4).to_string(index=False))
    print(f"\n  Tables → {tbl_dir}")

    # Save a JSON record of the run
    (tbl_dir / "run_config.json").write_text(json.dumps({
        "splits": args.splits, "models": models, "seeds": args.seeds,
        "n_samples": len(df), "n_folds": 5,
    }))
    return agg


if __name__ == "__main__":
    main()