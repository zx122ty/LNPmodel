"""Stage 3 — uncertainty quantification: GPR + conformal + quantile forest.

Competing approaches for calibrated prediction intervals on log10 transfection:

  1. Gaussian Process (GPR) with fold-internal kernel-tuning — returns
     predictive std; tune a scalar multiplier on calibration residuals.
  2. Split-conformal calibration on GPR point predictions (distribution-free
     guarantee, independent of GP's std quality).
  3. Quantile Regression Forest (QRF) — lightweight tree-based quantile
     intervals for comparison.

For each method we report, in log10 space:
    coverage @ 90%, avg interval width, MV-width / NLIW, and a small uncertainty
    plot of the best method.

All extrinsics (indices) come from the same dataset used in Stage 2; the random
80/10/10 split provides train/cal/test disjointness.
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


_N_FEATURES = 60   # GP is limited to ~50–60 selected features for tractability
_QRF_MODEL = "lightgbm"  # tree-based quantile baseline


def _build_data():
    """Load clean df + full feature matrix (for the given split)."""
    """Load clean df + full feature matrix (for the given split)."""
    from lnp.data.loader import load_dataset
    from lnp.features.engine import featurize
    from sklearn.impute import SimpleImputer

    df = load_dataset()
    X = featurize(df["smiles"].tolist())
    # column pruning like Stage 2: drop all-NaN cols
    X = X.dropna(axis=1, how="all")
    arr = X.to_numpy(float)
    # global mask extreme values (same threshold logic as engine)
    X = X.mask(pd.DataFrame(np.abs(arr) > 1e6, index=X.index, columns=X.columns))
    X = X.dropna(axis=1, how="all")
    # field-fill median (fit inside folds, here on full set only for indexing milestone)
    imp = SimpleImputer(strategy="median").fit(X)
    X = pd.DataFrame(imp.transform(X), columns=X.columns)
    return df, X


def split_cv(df, X, test_ratio=0.1, val_ratio=0.1, seed=42):
    """Random split into train/cal/test (indices)."""
    from lnp.splits.splitters import RandomSplitter

    sp = RandomSplitter(val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    s = sp.split(df["smiles"].tolist())
    # Use test as held-out; cal = val slice (pool from train+val minus test for best coverage)
    return s


def run_gpr_conformal(X, y, tr_idx, cal_idx, te_idx, alpha=0.05):
    """GPR + fold-internal static, calibrate std-multiplier / conformal."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from lnp.models.conformal import conformal_residual_quantile, apply_intervals

    X_tr, X_cal, X_te = X.iloc[tr_idx], X.iloc[cal_idx], X.iloc[te_idx]
    y_tr, y_cal, y_te = y[tr_idx], y[cal_idx], y[te_idx]

    # Sub-select features for GP (top-k by |F-ratio| to keep GP tractable).
    from sklearn.feature_selection import f_regression, SelectKBest
    sel = SelectKBest(f_regression, k=_N_FEATURES).fit(X_tr, y_tr)
    X_tr_k = sel.transform(X_tr)
    X_cal_k = sel.transform(X_cal)
    X_te_k = sel.transform(X_te)

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
    gpr.fit(X_tr_k, y_tr)

    mu_cal, std_cal = gpr.predict(X_cal_k, return_std=True)
    mu_te, std_te = gpr.predict(X_te_k, return_std=True)
    y_pred = mu_te
    mu_cal_arr = mu_cal

    # (a) Conformal additive interval on |residual|
    resid = np.abs(mu_cal_arr - y_cal)
    q = conformal_residual_quantile(resid, alpha=alpha)
    lo_c, hi_c = apply_intervals(y_pred, q)

    # (b) GP std-based, tuned multiplier on calibration to hit ~1-alpha coverage
    z = np.percentile(np.abs(mu_cal_arr - y_cal) / np.maximum(std_cal, 1e-9), 100 * (1 - alpha))
    lo_g = y_pred - z * std_te
    hi_g = y_pred + z * std_te

    return {
        "pred": y_pred,
        "lo_conformal": lo_c, "hi_conformal": hi_c, "q": q,
        "lo_gp": lo_g, "hi_gp": hi_g, "z": z,
        "std": std_te,
    }


def run_quantile_forest(X, y, tr_idx, cal_idx, te_idx, alpha=0.05):
    """Quantile Regression Forest via LightGBM quantile objective (two models)."""
    import lightgbm as lgb

    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr = y[tr_idx]
    base = dict(n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31,
                verbose=-1, random_state=42)
    lower_m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **base)
    upper_m = lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha, **base)
    lower_m.fit(X_tr, y_tr)
    upper_m.fit(X_tr, y_tr)
    lo = lower_m.predict(X_te)
    hi = upper_m.predict(X_te)
    return {"pred": (lo + hi) / 2, "lo": lo, "hi": hi}


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def mv_width(y, lo, hi):
    """Mean interval width normalised by |y| (NLIW-like)."""
    return float(np.mean((hi - lo) / np.maximum(np.abs(y), 1e-6)))


def main():
    parser = argparse.ArgumentParser(description="Stage 3 — uncertainty")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    df, X = _build_data()
    y = df["target"].to_numpy(float)
    print(f"[Stage 3] uncertainty — n={len(df)}, alpha={args.alpha}")

    rows = []
    oof_gpr = pd.DataFrame({"name": df["name"], "actual": y})
    for seed in args.seeds:
        s = split_cv(df, X, seed=seed)
        tr, cal, te = s.train, s.val, s.test
        res_g = run_gpr_conformal(X, y, tr, cal, te, alpha=args.alpha)
        res_q = run_quantile_forest(X, y, tr, cal, te, alpha=args.alpha)

        # evaluate on test
        y_te = y[te]

        rec = {"seed": seed, "alpha": args.alpha}
        rec["gpr_conformal_cov"] = coverage(y_te, res_g["lo_conformal"], res_g["hi_conformal"])
        rec["gpr_conformal_width"] = float(np.mean(res_g["hi_conformal"] - res_g["lo_conformal"]))
        rec["gpr_conformal_nliw"] = mv_width(y_te, res_g["lo_conformal"], res_g["hi_conformal"])
        rec["gpr_gp_cov"] = coverage(y_te, res_g["lo_gp"], res_g["hi_gp"])
        rec["gpr_gp_width"] = float(np.mean(res_g["hi_gp"] - res_g["lo_gp"]))
        rec["qrf_cov"] = coverage(y_te, res_q["lo"], res_q["hi"])
        rec["qrf_width"] = float(np.mean(res_q["hi"] - res_q["lo"]))

        rows.append(rec)
        # store predictions on test set (first seed)
        if seed == args.seeds[0]:
            for k in ["pred", "lo_conformal", "hi_conformal", "lo_gp", "hi_gp", "std"]:
                oof_gpr[f"{k}"] = np.nan
            oof_gpr.loc[te, "pred"] = res_g["pred"]
            oof_gpr.loc[te, "lo_conformal"] = res_g["lo_conformal"]
            oof_gpr.loc[te, "hi_conformal"] = res_g["hi_conformal"]
            oof_gpr.loc[te, "lo_gp"] = res_g["lo_gp"]
            oof_gpr.loc[te, "hi_gp"] = res_g["hi_gp"]
            oof_gpr.loc[te, "std"] = res_g["std"]

    rec_df = pd.DataFrame(rows)
    tbl_dir = config.TABLE_DIR / "03_uncertainty"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    rec_df.to_csv(tbl_dir / "uncertainty_summary.csv", index=False)

    # Figure: pred-vs-actual with conformal interval bars on the test slice
    from lnp.eval.visualize import plot_predicted_vs_actual

    fig_dir = config.FIG_DIR / "03_uncertainty"
    fig_dir.mkdir(parents=True, exist_ok=True)
    va = oof_gpr.dropna(subset=["actual", "pred", "lo_conformal", "hi_conformal"])
    plot_predicted_vs_actual(
        va["actual"], va["pred"],
        target_name="log10(transfection)",
        fname="pred_vs_actual_gpr_conformal",
        out_dir=fig_dir,
        y_lower=va["lo_conformal"], y_upper=va["hi_conformal"],
        title="GPR + split-conformal 90% intervals (test slice)",
    )

    print("\n  === Uncertainty summary (90% target) ===")
    print(rec_df.mean(numeric_only=True).round(4).to_string())
    print(f"\n  Tables → {tbl_dir}")
    return rec_df


if __name__ == "__main__":
    main()