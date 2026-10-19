"""Visualisation helpers for the transfection benchmark (publication-quality)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_CODE = Path(__file__).resolve().parents[3]  # benchmark/code/
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": config.FIG_DPI,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.figsize": (10, 7),
})


def save_fig(fig, name: str, out_dir=None) -> Path:
    out_dir = out_dir or config.FIG_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{config.FIG_FORMAT}"
    fig.savefig(path, bbox_inches="tight", dpi=config.FIG_DPI)
    plt.close(fig)
    return path


def plot_predicted_vs_actual(y_true, y_pred, target_name="log10 efficiency",
                             fname="pred_vs_actual", out_dir=None,
                             y_lower=None, y_upper=None, title=""):
    """Scatter of predicted vs actual, optionally with calibrated interval bars.

    If ``y_lower``/``y_upper`` are given, vertical errorbars are drawn to
    visualise the uncertainty bands (log10 space).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=28, edgecolors="k", linewidths=0.4, zorder=3)

    if y_lower is not None and y_upper is not None:
        ax.errorbar(y_true, y_pred,
                    yerr=[y_pred - np.asarray(y_lower), np.asarray(y_upper) - y_pred],
                    fmt="none", ecolor="gray", alpha=0.25, elinewidth=0.6, zorder=1)

    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    margin = (hi - lo) * 0.05
    lims = [lo - margin, hi + margin]
    ax.plot(lims, lims, "r--", alpha=0.7, linewidth=1, label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(title or f"Predicted vs Actual: {target_name}")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    return save_fig(fig, fname, out_dir=out_dir)


def plot_model_comparison(df, metric="R2", ylabel="R² (mean)",
                          title="Model Performance Comparison", fname="model_comparison",
                          out_dir=None, split=None):
    """Grouped bar chart comparing a metric across models.

    ``df`` is an aggregated summary (already mean/std collapsed, one row per
    model). ``metric`` names a column holding the plotted value.
    """
    models = df["model"].astype(str).tolist()
    vals = df[metric].astype(float).tolist()
    std_col = [c for c in df.columns if c.startswith(metric) and "std" in c.lower()]
    errs = df[std_col[0]].astype(float).tolist() if std_col else None

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.1), 6))
    x = np.arange(len(models))
    ax.bar(x, vals, yerr=errs, capsize=3, alpha=0.85, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return save_fig(fig, fname, out_dir=out_dir)


def plot_split_comparison(summary_df, metrics=("R2", "RMSE", "Spearman_rho"), out_dir=None):
    """Groups bars by model, coloured by split method (random/scaffold/cluster).

    ``summary_df`` has one row per (split, model) with scalar metric columns.
    """
    models = summary_df["model"].astype(str).unique()
    splits = summary_df["split"].astype(str).unique()
    palette = plt.get_cmap("Set2")(np.linspace(0, 1, max(len(splits), 1)))

    nrows = len(metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    width = 0.8 / len(splits)

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(models))
        for j, sp in enumerate(splits):
            sub = summary_df[summary_df["split"].astype(str) == sp].set_index("model")
            vals = [sub.loc[m, metric] if m in sub.index else np.nan for m in models]
            ax.bar(x + j * width, vals, width, label=str(sp), color=palette[j], alpha=0.9, capsize=2)
        ax.set_xticks(x + width * (len(splits) - 1) / 2)
        ax.set_xticklabels([str(m) for m in models], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        if metric == "RMSE":
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8)
    axes[0].set_title("Model performance across split strategies")
    fig.tight_layout()
    return save_fig(fig, "split_comparison", out_dir=out_dir)