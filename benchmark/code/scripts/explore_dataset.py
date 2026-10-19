"""Stage 0 — Exploratory data analysis (EDA).

Produces the initial data snapshot the user reviews before modelling:

  * summary statistics of the target (raw + log10)
  * SMILES parsing / dedup handling report
  * target distribution histogram + KDE
  * high-level structural/activity cluster view (per-table + plot)

Runs on ``dataset.xlsx/Sheet3`` only. Writes figures and a report CSV under
``outputs/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1]  # benchmark/code/
sys.path.insert(0, str(PROJECT_CODE / "src"))
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402


def _save(name: str, fig=None, ext: str | None = None) -> Path:
    out_dir = config.FIG_DIR / "00_eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{ext or config.FIG_FORMAT}"
    if fig is not None:
        fig.savefig(path, bbox_inches="tight", dpi=config.FIG_DPI)
        plt.close(fig)
    return path


def plot_target_distribution(df: pd.DataFrame) -> Path:
    raw = df["target_raw"].to_numpy(float)
    logt = df["target"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # raw scale (log-x to visualise wide dynamic range)
    axes[0].hist(raw, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Raw transfection efficiency (log axis)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Raw target  (n={len(raw)})")

    # log10 scale
    axes[1].hist(logt, bins=60, color="darkorange", alpha=0.75, edgecolor="white")
    axes[1].set_xlabel("log10(transfection efficiency)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Transformed target (modeled)")
    fig.tight_layout()
    return _save("target_distribution", fig)


def cluster_activity_view(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Butina-cluster lipids and report mean log10 activity per cluster.

    Reinforces the Stage-0 objective (structure cluster ↔ transfection rate).
    Returns a DataFrame of cluster stats for saving.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.ML.Cluster import Butina

    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols if m is not None]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]

    num = len(fps)
    if num < 2:
        return pd.DataFrame()
    dists = []
    for i in range(num):
        for j in range(i + 1, num):
            dists.append(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    clusters = [list(c) for c in Butina.ClusterData(dists, num, 0.3, isDistData=True)]

    rows = []
    solos = 0
    for cid, cluster in enumerate(clusters):
        idxs = [valid_idx[pos] for pos in cluster]
        y = df.iloc[idxs]["target"].to_numpy()
        rows.append({
            "cluster_id": cid,
            "size": len(idxs),
            "mean_log10": float(np.mean(y)),
            "std_log10": float(np.std(y)),
            "min_log10": float(np.min(y)),
            "max_log10": float(np.max(y)),
            "example_name": df.iloc[idxs[0]]["name"],
        })
        if len(cluster) == 1:
            solos += 1
    out = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(out["size"], out["mean_log10"], c="indianred", alpha=0.6)
    axes[0].set_xlabel("Cluster size")
    axes[0].set_ylabel("Mean log10 transfection / cluster")
    axes[0].set_title(f"{num} clustered lipids, {len(out)} clusters, {solos} singletons")
    # top-12 most active clusters
    top = out.sort_values("mean_log10", ascending=False).head(12)
    axes[1].barh(np.arange(len(top))[::-1], top["mean_log10"], color="teal")
    axes[1].set_yticks(np.arange(len(top))[::-1])
    axes[1].set_yticklabels(top["example_name"], fontsize=7)
    axes[1].set_xlabel("Mean log10")
    axes[1].set_title("Most active clusters (by mean log10)")
    fig.tight_layout()
    _save("cluster_activity_vs_log10", fig)
    return out


def run() -> pd.DataFrame:
    print("[Stage 0] Exploratory data analysis...")
    from lnp.data.loader import load_dataset

    df = load_dataset()
    report = df.attrs["report"]
    print(f"  raw rows={report['raw_rows']} invalid_smiles={report.get('invalid_smiles')} "
          f"dedup_dropped={report.get('dup_smiles_dropped', 0)} final_rows={report['final_rows']}")

    # Save processed snapshot
    tbl_dir = config.TABLE_DIR / "00_eda"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    df[["id", "name", "smiles", "target_raw", "target"]].to_csv(
        tbl_dir / "cleaned_snapshot.csv", index=False)

    # Target dist plots + report table
    plot_target_distribution(df)
    summary = pd.DataFrame({
        "metric": ["n_samples", "target_min_log10", "target_max_log10",
                   "target_mean_log10", "target_std_log10"],
        "value": [len(df), df["target"].min(), df["target"].max(),
                  df["target"].mean(), df["target"].std()],
    })
    summary.to_csv(tbl_dir / "target_summary.csv", index=False)

    # Cluster-activity
    cluster_stats = cluster_activity_view(df)
    if not cluster_stats.empty:
        cluster_stats.to_csv(tbl_dir / "cluster_activity.csv", index=False)

    print("  EDA complete → figures saved.")
    return df


if __name__ == "__main__":
    run()