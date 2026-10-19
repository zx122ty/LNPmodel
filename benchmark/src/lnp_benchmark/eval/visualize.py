"""
visualize.py — Visualization suite for the LNP Benchmark Pipeline.

Generates 4 types of plots, grouped by split method (random / scaffold / cluster):

  1. Grouped bar chart + error bars — RMSE and MAE per model, bars = seeds,
     error bars = std across seeds
  2. Heatmap — R², Spearman, Pearson mean across seeds
  3. Radar chart — normalised multi-metric (R², Spearman, 1/RMSE, 1/MAE)
  4. Scatter plot — predicted vs true for the best model (by R²)

All settings are controllable via a config dict for extensibility.
"""

import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Default plot config ─────────────────────────────────────────────────────

DEFAULT_PLOT_CONFIG = {
    "figsize_large": (14, 7),
    "figsize_medium": (10, 8),
    "figsize_small": (8, 6),
    "dpi": 150,
    "bar_alpha": 0.85,
    "errorbar_capsize": 4,
    "heatmap_cmap": "RdBu_r",
    "heatmap_annot_fontsize": 10,
    "radar_alpha": 0.15,
    "scatter_alpha": 0.6,
    "scatter_s": 20,
    "title_fontsize": 14,
    "label_fontsize": 11,
    "tick_fontsize": 9,
    "legend_fontsize": 9,
}


# ── Public API ──────────────────────────────────────────────────────────────

def generate_all_plots(
    raw_csv: str,
    agg_csv: str,
    output_dir: str,
    plot_config: Optional[Dict] = None,
) -> List[str]:
    """
    Generate all 4 chart types, grouped by split method.

    Parameters
    ----------
    raw_csv : str
        Path to raw_results.csv.
    agg_csv : str
        Path to aggregated_results.csv.
    output_dir : str
        Directory to save the generated PNG files.
    plot_config : dict, optional
        Overrides for DEFAULT_PLOT_CONFIG.

    Returns
    -------
    List of saved file paths.
    """
    cfg = {**DEFAULT_PLOT_CONFIG, **(plot_config or {})}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_agg = pd.read_csv(agg_csv)
    df_raw = pd.read_csv(raw_csv)

    # ── Determine split groups from the data ────────────────────────────
    # split_name could be "random" (base) or "random_seed42" (detailed)
    # Map to base method
    def _base_split(name: str) -> str:
        return re.sub(r"_seed\d+$", "", name)

    # Add a base_split column for grouping
    df_agg["_base_split"] = df_agg["split_name"].apply(_base_split)
    df_raw["_base_split"] = df_raw["split_name"].apply(_base_split)
    split_groups = sorted(df_agg["_base_split"].unique())

    saved_files = []
    for base_split in split_groups:
        # Filter to this split group
        agg_sub = df_agg[df_agg["_base_split"] == base_split].copy()
        raw_sub = df_raw[df_raw["_base_split"] == base_split].copy()

        if agg_sub.empty:
            continue

        # Determine model order by mean R² across seeds
        if "R2_mean" in agg_sub.columns:
            model_order = (
                agg_sub.groupby("model_name")["R2_mean"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
        else:
            model_order = agg_sub["model_name"].unique().tolist()

        # ── 1. Bar chart + error bars ───────────────────────────────────
        fig1 = _plot_bar_error(agg_sub, base_split, cfg, model_order)
        path1 = out / f"bar_error_{base_split}.png"
        fig1.savefig(path1, dpi=cfg["dpi"], bbox_inches="tight")
        plt.close(fig1)
        saved_files.append(str(path1))

        # ── 2. Heatmap ──────────────────────────────────────────────────
        if all(c in agg_sub.columns for c in ["R2_mean", "Spearman_mean", "Pearson_mean"]):
            fig2 = _plot_heatmap(agg_sub, base_split, cfg, model_order)
            path2 = out / f"heatmap_{base_split}.png"
            fig2.savefig(path2, dpi=cfg["dpi"], bbox_inches="tight")
            plt.close(fig2)
            saved_files.append(str(path2))

        # ── 3. Radar chart ──────────────────────────────────────────────
        if all(c in agg_sub.columns for c in ["R2_mean", "Spearman_mean", "RMSE_mean", "MAE_mean"]):
            fig3 = _plot_radar(agg_sub, base_split, cfg, model_order)
            path3 = out / f"radar_{base_split}.png"
            fig3.savefig(path3, dpi=cfg["dpi"], bbox_inches="tight")
            plt.close(fig3)
            saved_files.append(str(path3))

        # ── 4. Scatter plot (best model per seed) ───────────────────────
        # raw_sub has R2 (not R2_mean), y_true, y_pred per seed
        if "R2" in raw_sub.columns and "y_true" in raw_sub.columns and "y_pred" in raw_sub.columns:
            fig4 = _plot_scatter(raw_sub, base_split, cfg)
            path4 = out / f"scatter_{base_split}.png"
            fig4.savefig(path4, dpi=cfg["dpi"], bbox_inches="tight")
            plt.close(fig4)
            saved_files.append(str(path4))

    return saved_files


def generate_all_plots_standalone(
    results_dir: str,
    plot_config: Optional[Dict] = None,
) -> List[str]:
    """
    Standalone entry point — run from command line or after pipeline.

    Usage:
        from lnp_benchmark.eval.visualize import generate_all_plots_standalone
        generate_all_plots_standalone("/path/to/results")
    """
    results_dir = Path(results_dir)
    raw_csv = results_dir / "raw_results.csv"
    agg_csv = results_dir / "aggregated_results.csv"
    out_dir = results_dir / "figures"

    if not raw_csv.exists():
        raise FileNotFoundError(f"raw_results.csv not found in {results_dir}")
    if not agg_csv.exists():
        raise FileNotFoundError(f"aggregated_results.csv not found in {results_dir}")

    return generate_all_plots(str(raw_csv), str(agg_csv), str(out_dir), plot_config)


# ── Chart implementations ───────────────────────────────────────────────────

def _plot_bar_error(
    df: pd.DataFrame,
    base_split: str,
    cfg: Dict,
    model_order: List[str],
) -> plt.Figure:
    """
    Grouped bar chart + error bars per model for RMSE and MAE.

    Each model gets one bar per seed, with error bars from _std columns.
    """
    metrics = ["RMSE", "MAE"]
    metric_labels = {"RMSE": "RMSE", "MAE": "MAE"}
    n_models = len(model_order)
    if n_models == 0:
        fig, ax = plt.subplots(figsize=cfg["figsize_small"])
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # Get unique seeds for this split group
    seeds = sorted(df["seed"].unique()) if "seed" in df.columns else [0]
    n_seeds = len(seeds)
    colors = plt.cm.Set2(np.linspace(0.2, 0.8, max(n_models, 1)))

    metric_col_map = {
        "RMSE": ("RMSE_mean", "RMSE_std"),
        "MAE": ("MAE_mean", "MAE_std"),
    }

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize_large"])
    for ax, metric in zip(axes, metrics):
        mean_col, std_col = metric_col_map[metric]
        if mean_col not in df.columns:
            ax.text(0.5, 0.5, f"{mean_col} not found", ha="center", va="center")
            continue

        # For each model, collect mean across seeds for bar + std for error
        bar_means = []
        bar_stds = []
        for model in model_order:
            sub = df[df["model_name"] == model]
            vals = sub[mean_col].values.astype(float)
            bar_means.append(np.mean(vals))
            bar_stds.append(np.std(vals) if len(vals) > 1 else 0.0)

        bar_means = np.array(bar_means)
        bar_stds = np.array(bar_stds)

        x = np.arange(n_models)
        bars = ax.bar(
            x, bar_means, yerr=bar_stds,
            capsize=cfg["errorbar_capsize"],
            color=colors[:n_models],
            alpha=cfg["bar_alpha"],
            edgecolor="gray", linewidth=0.5,
        )

        # Annotate
        for bar, val in zip(bars, bar_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2e}",
                ha="center", va="bottom",
                fontsize=cfg["tick_fontsize"] - 1,
                rotation=45,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=30, ha="right",
                           fontsize=cfg["tick_fontsize"])
        ax.set_ylabel(metric_labels[metric], fontsize=cfg["label_fontsize"])
        ax.set_title(f"{metric_labels[metric]} — {base_split}",
                     fontsize=cfg["title_fontsize"])

    fig.suptitle(f"Model Comparison — {base_split} split",
                 fontsize=cfg["title_fontsize"] + 2, y=1.02)
    fig.tight_layout()
    return fig


def _plot_heatmap(
    df: pd.DataFrame,
    base_split: str,
    cfg: Dict,
    model_order: List[str],
) -> plt.Figure:
    """Heatmap of R², Spearman, Pearson mean values (averaged across seeds)."""
    metric_cols = ["R2_mean", "Spearman_mean", "Pearson_mean"]
    metric_labels = ["R²", "Spearman ρ", "Pearson r"]

    # Aggregate across seeds
    agg = df.groupby("model_name")[metric_cols].mean()
    # Reorder
    agg = agg.reindex(model_order)
    values = agg.values.astype(float)

    fig, ax = plt.subplots(figsize=cfg["figsize_medium"])
    cmap = plt.cm.get_cmap(cfg["heatmap_cmap"], 50)
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if abs(values[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{values[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=cfg["heatmap_annot_fontsize"],
                    color=color)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=cfg["label_fontsize"])
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=cfg["tick_fontsize"])
    ax.set_title(f"Correlation Metrics — {base_split} split",
                 fontsize=cfg["title_fontsize"])
    plt.colorbar(im, ax=ax, shrink=0.8, label="Score")
    fig.tight_layout()
    return fig


def _plot_radar(
    df: pd.DataFrame,
    base_split: str,
    cfg: Dict,
    model_order: List[str],
) -> plt.Figure:
    """
    Radar chart with normalised metrics.

    Metrics: R², Spearman, 1/RMSE, 1/MAE — all min-max scaled to [0, 1].
    Values are averaged across seeds.
    """
    # Aggregate across seeds
    agg = df.groupby("model_name").agg({
        "R2_mean": "mean",
        "Spearman_mean": "mean",
        "RMSE_mean": "mean",
        "MAE_mean": "mean",
    }).reindex(model_order)

    raw_metrics = {
        "R²": agg["R2_mean"].values.astype(float),
        "Spearman ρ": agg["Spearman_mean"].values.astype(float),
        "1 / RMSE": 1.0 / (agg["RMSE_mean"].values.astype(float) + 1e-10),
        "1 / MAE": 1.0 / (agg["MAE_mean"].values.astype(float) + 1e-10),
    }
    labels = list(raw_metrics.keys())
    n_metrics = len(labels)
    n_models = len(model_order)

    # Min-max normalise
    normalised = {}
    for k, v in raw_metrics.items():
        vmin, vmax = v.min(), v.max()
        if vmax - vmin > 1e-12:
            normalised[k] = (v - vmin) / (vmax - vmin)
        else:
            normalised[k] = np.ones_like(v)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=cfg["figsize_small"],
                           subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0.1, 0.9, max(n_models, 1)))

    for i in range(n_models):
        values = [normalised[k][i] for k in labels]
        values += values[:1]
        ax.plot(angles, values, "o-", label=model_order[i],
                color=colors[i], linewidth=1.5, markersize=4)
        ax.fill(angles, values, alpha=cfg["radar_alpha"], color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=cfg["label_fontsize"])
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"],
                       fontsize=cfg["tick_fontsize"] - 2)
    ax.set_title(f"Normalised Metrics — {base_split} split",
                 fontsize=cfg["title_fontsize"], pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=cfg["legend_fontsize"])
    fig.tight_layout()
    return fig


def _plot_scatter(
    df: pd.DataFrame,
    base_split: str,
    cfg: Dict,
) -> plt.Figure:
    """
    Predicted vs true scatter plot for the best model (by R²).

    Requires raw_results columns: model_name, R2, y_true, y_pred.
    """
    required = {"model_name", "R2", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        fig, ax = plt.subplots(figsize=cfg["figsize_small"])
        ax.text(0.5, 0.5,
                "y_true / y_pred not in raw data.\n"
                "Set output.save_predictions=true in config.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Find best model by mean R² across seeds
    best_model = df.groupby("model_name")["R2"].mean().idxmax()
    sub = df[df["model_name"] == best_model]

    # Collect all y_true, y_pred for this model across seeds
    all_true = []
    all_pred = []
    for _, row in sub.iterrows():
        yt = row["y_true"]
        yp = row["y_pred"]
        if isinstance(yt, str):
            yt = np.fromstring(yt.strip("[]"), sep=",")
        if isinstance(yp, str):
            yp = np.fromstring(yp.strip("[]"), sep=",")
        if isinstance(yt, (np.ndarray, list)) and not np.isscalar(yt):
            all_true.extend(np.asarray(yt, dtype=float).tolist())
            all_pred.extend(np.asarray(yp, dtype=float).tolist())

    if not all_true:
        fig, ax = plt.subplots(figsize=cfg["figsize_small"])
        ax.text(0.5, 0.5, "Per-point predictions not available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    spr, _ = spearmanr(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=cfg["figsize_small"])
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.scatter(y_true, y_pred, alpha=cfg["scatter_alpha"],
               s=cfg["scatter_s"], edgecolors="gray", linewidth=0.3)
    ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Value", fontsize=cfg["label_fontsize"])
    ax.set_ylabel("Predicted Value", fontsize=cfg["label_fontsize"])
    ax.set_title(f"{best_model} — {base_split}\n"
                 f"R²={r2:.4f}  Spearman={spr:.4f}  Pearson={pcc:.4f}",
                 fontsize=cfg["title_fontsize"])
    ax.legend(fontsize=cfg["legend_fontsize"])
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig