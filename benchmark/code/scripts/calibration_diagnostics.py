"""Calibration diagnostic figures for the final conformal interval.

Produces qualitative diagnostics to show the interval is well-behaved where it
matters and honestly wider where the model is unsure:

  1. Interval width vs predicted magnitude — is the band constant (that's what
     a scalar conformal q gives) or does it fan out (heteroscedastic)? We show
     it as a diagnostic; conformal q is scalar, so expect a flat band.
     If providers want local width later, this reveals where.
  2. Coverage-by-bin: slice test points into width bins and confirm coverage
     is roughly flat at the target across the range.
  3. Calibration residuals histogram vs the fitted q (where a scalar q sits).

Uses the stored point model + conformal q against the benchmark's own hold-out
slice so we can show both residual fit and true labels. Output PNGs under
outputs/figures/05_calibration/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1]  # benchmark/code
sys.path.insert(0, str(PROJECT_CODE / "src"))
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402

from lnp.models_final import MODEL_DIR, META_FILE, QFILE, predict_with_interval


def _titles(fname, fig):
    fig.tight_layout()
    out = config.FIG_DIR / "05_calibration"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{fname}.png", bbox_inches="tight", dpi=config.FIG_DPI)
    plt.close(fig)
    return out / f"{fname}.png"


def main():
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

    # Hold out a slice to evaluate calibration against true labels.
    _, X_eval, _, y_eval = train_test_split(X, y, test_size=0.2, random_state=7)
    meta = json.loads((MODEL_DIR / META_FILE).read_text())
    q = json.loads((MODEL_DIR / QFILE).read_text())["q"]

    # align to stored feature names
    cols = meta["feature_names"]
    Xe = pd.DataFrame(0.0, index=X_eval.index, columns=cols)
    for c in cols:
        if c in X_eval.columns:
            Xe[c] = X_eval[c].values

    pred, lo, hi, _ = predict_with_interval(Xe)
    res = np.abs(pred - y_eval)
    width = hi - lo  # = 2*q, constant for scalar conformal

    # 1. Interval width vs predicted magnitude
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred, width, alpha=0.5, s=22, c="steelblue")
    ax.axhline(2 * q, color="crimson", ls="--", lw=1.5, label=f"scalar conformal width = 2q ({2*q:.2f})")
    ax.set_xlabel("log10 predicted transfection")
    ax.set_ylabel("90% interval width (log10)")
    ax.set_title("Interval width vs predicted magnitude (scalar conformal → flat)")
    ax.legend()
    _titles("width_vs_pred", fig)

    # 2. Coverage by bin (quantile bins of predicted magnitude)
    bins = pd.qcut(pred, 4, duplicates="drop")
    cov_by_bin = pd.DataFrame({
        "bin": bins,
        "covered": (y_eval >= lo) & (y_eval <= hi),
    }).groupby("bin", observed=True)["covered"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xpos = np.arange(len(cov_by_bin))
    ax.bar(xpos, cov_by_bin["mean"].values * 100, color="teal", alpha=0.75)
    ax.axhline(90, color="crimson", ls="--", lw=1.5, label="target 90%")
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(i) for i in cov_by_bin.index], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("coverage (%)")
    ax.set_xlabel("predicted-magnitude bin")
    ax.set_title(f"Coverage-by-bin (n={len(y_eval)} eval pts) — flat ≈ target = well calibrated")
    ax.legend()
    _titles("coverage_vs_pred_bin", fig)

    # 3. Residual histogram vs fitted q
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(res, bins=40, alpha=0.6, color="steelblue", density=True)
    ax.axvline(q, color="crimson", ls="--", lw=2, label=f"conformal q = {q:.3f}")
    ax.axvline(np.percentile(res, 90), color="darkorange", ls=":", lw=2,
               label=f"empirical 90th = {np.percentile(res, 90):.3f}")
    ax.set_xlabel("|pred − y| residuals (log10)")
    ax.set_ylabel("density")
    ax.set_title("Evaluation |residual| vs fitted conformal quantile")
    ax.legend()
    _titles("residual_vs_q", fig)

    # print table summary
    print("[calibration] figures written → outputs/figures/05_calibration/")
    print(f"  eval pts={len(y_eval)}  overall coverage={(np.mean((y_eval>=lo)&(y_eval<=hi))*100):.1f}%  "
          f"mean width={width.mean():.3f}")
    print(cov_by_bin[["mean", "count"]].round(3).to_string())


if __name__ == "__main__":
    main()