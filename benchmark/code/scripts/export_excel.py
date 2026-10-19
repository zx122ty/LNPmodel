"""Write benchmark results to a multi-sheet Excel workbook.

Consolidates Stages 0/2/3/5 outputs into outputs/lnp_benchmark_results.xlsx.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_CODE = Path(__file__).resolve().parents[1]  # benchmark/code
sys.path.insert(0, str(PROJECT_CODE / "src"))
sys.path.insert(0, str(PROJECT_CODE))
import config  # noqa: E402

OUTPUT_XLSX = config.OUTPUT_DIR / "lnp_benchmark_results.xlsx"


def _read(path):
    return pd.read_csv(path) if path.exists() else None


def _pick(df, keep):
    if df is None:
        return df
    cols = [c for c in keep if c in df.columns]
    return df[cols]


def main():
    sheets = {}
    sheets["predictions"] = _read(config.TABLE_DIR / "predictions.csv")

    base = _read(config.TABLE_DIR / "02_baseline" / "baseline_summary.csv")
    sheets["baseline"] = _pick(base, [
        "split", "model", "R2_mean", "R2_std", "RMSE_mean", "RMSE_std",
        "MAE_mean", "MAE_std", "Spearman_mean", "Spearman_std",
        "Pearson_mean", "Pearson_std"])

    sheets["uncertainty"] = _read(config.TABLE_DIR / "03_uncertainty" / "uncertainty_summary.csv")
    sheets["structural_summary"] = _read(config.TABLE_DIR / "04_structural" / "structural_summary.csv")
    sheets["leakage_diagnostic"] = _read(config.TABLE_DIR / "04_structural" / "split_leakage_diagnostic.csv")
    sheets["eda_target_summary"] = _read(config.TABLE_DIR / "00_eda" / "target_summary.csv")
    sheets["eda_cleaned_snapshot"] = _read(config.TABLE_DIR / "00_eda" / "cleaned_snapshot.csv")
    sheets["eda_cluster_activity"] = _read(config.TABLE_DIR / "00_eda" / "cluster_activity.csv")

    cfg_rows = [{"key": k, "value": v} for k, v in {
        "dataset": str(config.DATASET_FILE), "sheet": config.DATASET_SHEET,
        "random_seed": config.RANDOM_SEED, "seeds": ",".join(map(str, config.SEEDS)),
        "base_models": ",".join(config.BASE_MODELS), "target_unit": config.TARGET_UNIT,
    }.items()]
    sheets["run_config"] = pd.DataFrame(cfg_rows)

    if OUTPUT_XLSX.exists():
        OUTPUT_XLSX.unlink()
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"  [xlsx] wrote → {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()