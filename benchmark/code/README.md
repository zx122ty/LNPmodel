# Ionizable Lipid Transfection Prediction (benchmark)

A slim, reproducible pipeline that predicts **log10 transfection efficiency**
for ionizable lipid nanoparticles (LNPs) from lipid **SMILES** and returns
**calibrated prediction intervals (uncertainty)**.

This is a clean new project under `benchmark/code/`, physically isolated from
the legacy multi-target pipeline in `Work/code/`. It reuses the
`descriptor_pipeline` package for molecular featurisation but is scoped to a
*single* regression target with conformal uncertainty.

## Data
- Source: `../input_data/dataset.xlsx` / sheet `Sheet3`
- 768 raw rows → **759** after deduplicating identical canonical SMILES (9 dropped)
- Target: `transfection_efficiency` (luminescence_normalized): raw 2.06e3 – 9.93e7 → modeled on `log10`

## Directory layout
```
benchmark/code/
├── config.py                 # all paths, seeds, model/split settings
├── src/lnp/
│   ├── data/loader.py        # load/clean/dedup + log10 target transform
│   ├── features/engine.py    # descriptor_pipeline wrapper
│   ├── descriptor_pipeline/  # reused RDKit/fingerprint/LNP feature engine
│   ├── splits/splitters.py   # Random / Scaffold (Bemis-Murcko) / Cluster (Butina)
│   ├── models/pipeline.py    # sklearn pipelines (fold-internal preprocessing)
│   ├── models/conformal.py   # split-conformal residual quantile
│   ├── models_final.py       # final artifact save/load
│   ├── eval/metrics.py       # regression + uncertainty metrics
│   └── eval/visualize.py     # plots
├── scripts/
│   ├── explore_dataset.py    # Stage 0 EDA
│   ├── train.py              # Stage 2 baseline (3 splits × models) + figures/CSV
│   ├── uncertainty.py        # Stage 3 GPR/QRF interval comparison
│   ├── structural_robustness.py # Stage 4 leakage + OOD degradation
│   ├── train_final.py        # Stage 5 fit serialized point model + conformal q
│   └── predict.py            # CLI: new SMILES → prediction ± 90% interval
└── outputs/                  # generated tables, figures, models
```

## Install
```bash
cd benchmark/code
pip install -r requirements.txt
```
Confirmed env: Python 3.12, RDKit 2026.3, scikit-learn 1.8, lightgbm 4.6, xgboost, joblib.

## Run

### One-command (reproducible) — recommended
```bash
cd benchmark/code
python run_all.py                 # full pipeline incl. optional MLP exploration
python run_all.py --skip-explore  # omit the optional DL/MLP comparison
python run_all.py --skip-uncertainty  # skip Stage 3 (faster smoke test)
python run_all.py --seed 123      # override the base seed (default 42)
```
`run_all.py` regenerates every deliverable end-to-end (Stages 0/2/3/4/5 + the
optional (a)/(b)/(c) extras below).

### Step-by-step
```bash
# 1. EDA snapshot
python scripts/explore_dataset.py

# 2. baseline models (random / scaffold / cluster)×models
python scripts/train.py

# 3. uncertainty (GPR + split-conformal, LightGBM QRF contrast)
python scripts/uncertainty.py

# 4. structural robustness / OOD degradation
python scripts/structural_robustness.py

# 5. fit + serialize final model & interval, then predict
python scripts/train_final.py
python scripts/predict.py --smiles "CCCCOC(=O)C(C)SCC(COC(=O)CCC)OC(=O)CCOC(=O)CCNCCO"
```
`predict.py` also accepts `--csv file.csv --smiles-col SMILES` to score a batch,
and `--no-backtransform` to keep log10 output only.

### Extras (a · b · c)
```bash
# (a) multi-sheet Excel deliverable consolidating all stage outputs
python scripts/export_excel.py           # → outputs/lnp_benchmark_results.xlsx

# (b) optional DL contrast — Deep-Ensemble MLP (torch-free). GNN auto-skipped
#     because torch/torch_geometric are not installed (keeps env lightweight).
python scripts/deep_ensemble_explore.py --members 10

# (c) interval calibration diagnostics (width-vs-pred, coverage-by-bin, residuals)
python scripts/calibration_diagnostics.py
```

## Outputs
- `outputs/tables/02_baseline/` — per-fold metrics, aggregated summary, OOF wide
- `outputs/tables/03_uncertainty/uncertainty_summary.csv` (coverage/width/NLIW)
- `outputs/tables/04_structural/` — split-robustness + leakage diagnostic
- `outputs/figures/…` — PNG (300 dpi)
- `outputs/models/final/` — `point_model.job`, `meta.json`, `conformal_q.json`
- `outputs/tables/predictions.csv` — scored new lipids (log10 + raw + interval)
- `outputs/figures/05_calibration/` — (c) interval calibration diagnostics
- `outputs/tables/06_explore/` + `outputs/figures/06_explore/` — (b) MLP-ensemble contrast
- `outputs/lnp_benchmark_results.xlsx` — (a) single Excel deliverable, sheets:
  `predictions · baseline · uncertainty · structural_summary · leakage_diagnostic ·
  eda_target_summary · eda_cleaned_snapshot · eda_cluster_activity · run_config`

## Key findings
- **Point prediction:** RandomForest dominates without OOD pressure: R²≈0.79,
  Spearman≈0.85 on random/cluster splits; XGBoost close.
- **OOD caveat:** this 759-lipid set is a tight homologous family (test–train
  max Tanimoto ≈ 0.99 for random/cluster, ≈0.73 even for scaffold). So high
  random-split scores reflect family interpolation, **not** extrapolation to
  novel scaffolds (scaffold R²<0). Do not read scaffold metrics as a rejection
  of the model — the data set cannot support true OOD validation.

- **Uncertainty:** split-conformal on the GP gave overly wide (±≈3 log10) bands
  vs the weaker GP base. Deployed a **RF point backbone + conformal residual q**
  (final q≈0.76 → 90% interval ≈ ±0.76 log10, coverage≈0.91, width≈1.5 log10),
  which is tighter and honest (cal ± disjoint from training).
  Calibration diagnostics confirm coverage ≈96.7% overall and 1.00/0.95/0.97/0.95
  across predicted-magnitude bins (flat, ≥ nominal → well calibrated).

- **(b) DL contrast:** a 10-member Deep-Ensemble MLP matches RF point accuracy
  (R²≈0.79, Spearman≈0.88) but its Gaussian-std 90% interval badly under-covers
  (≈0.48). This supports the choice of a tree point model + distribution-free
  conformal band. GNN is only invoked if torch/torch_geometric are installed
  (they aren't here, so it is skipped to keep the env lightweight).

## Reproducibility
All seeds pinned in `config.py` (`RANDOM_SEED=42`, multi-seed list `SEEDS`).
All feature preprocessing (VarianceThreshold / impute-median / SelectKBest /
FP-PCA) is **inside** each model's sklearn pipeline → fit only on training folds,
no CV leakage. See `config.py` for the single source of truth on paths/seeds.