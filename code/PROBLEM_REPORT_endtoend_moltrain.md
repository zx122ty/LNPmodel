# PROBLEM REPORT: LipidFM Downstream Fine-tuning — End-to-End MolTrain Produces Catastrophically Bad R²

**Date:** 2026-08-13
**Author:** Claude (handoff for the next agent to fix)
**Status:** Open — needs diagnosis and fix

---

## 1. Project Context

This is the **LipidFM** project: a domain-adapted 3D molecular foundation model for **ionizable lipids (ILs)** used in LNP (lipid nanoparticle) delivery.

- **Upstream:** `Uni-Mol2` (84M) was continued-pretrained on ~133k ionizable-lipid molecules (12,381 real literature lipids + 100,000 enumerated virtual lipids) via `01_continous_training/4_scripts/run_lipids_continue_pretrain.py` (30k steps, `unicore-train --finetune-from-model checkpoint_84M.pt`). Result: the **LipidFM v2** checkpoint.
- **Downstream:** Two transfection-efficiency regression datasets are used to evaluate whether LipidFM beats the stock Uni-Mol2 base.
- **Goal:** Fine-tune with `unimol_tools.MolTrain(task='regression', ...)` end-to-end and compare LipidFM vs. the stock 84M base.

---

## 2. The Core Problem

**End-to-end MolTrain fine-tuning on our two downstream datasets produces catastrophically negative R² (≈ -100 to -2e45), while frozen-feature evaluation on the same checkpoint and data gives healthy R² (≈ 0.57–0.62).** The stock 84M base, when fine-tuned by the *official* Uni-Mol tutorial (see §4), gives R² ≈ 0.79.

This strongly suggests the problem is in **how we are invoking/reading `MolTrain`**, not in the model weights or the data.

### Evidence table

| Method | Dataset | R² | Notes |
|---|---|---|---|
| Frozen features + Ridge (`feature_benchmark.py`) | Sheet2 | **0.616** | ✅ healthy |
| Frozen features + Ridge (`feature_benchmark.py`) | Sheet3 (log10) | **0.568** | ✅ healthy |
| Official Uni-Mol tutorial `MolTrain` (hERG, stock 84M) | its own test split | **0.787** | ✅ healthy (their setup works) |
| Our `finetune_moltrain.py` end-to-end (LipidFM v2) | Sheet2 | **-110.98** (single fold, lr=5e-5) | ❌ broken |
| Our `finetune_moltrain.py` end-to-end (LipidFM v2) | Sheet2 | **-1271** (5-fold, lr=1e-4) | ❌ broken |
| Our `finetune_moltrain.py` end-to-end (LipidFM v2) | Sheet3 | **-2e100** | ❌ catastrophically broken |

**The checkpoint itself is verified good:** frozen features from `checkpoint_best.pt` load cleanly (502 backbone keys; only `classification_head.*` missing, which is expected for a pretrain-only checkpoint) and produce healthy R².

---

## 3. Data & Paths

### Datasets (one xlsx, two sheets)
`/home/zx/df/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx`

| Sheet | target column | samples | scale |
|---|---|---|---|
| Sheet2 | `RLU (log2)` | 1920 | already log2 |
| Sheet3 | `transfection_efficiency` | 768 | raw RLU, spans ~4 orders of magnitude → must `log10` before modeling |

### Checkpoints
- Stock Uni-Mol2 84M base: `01_continous_training/2_models/pretrained/checkpoint_84M.pt` (simple state-dict format)
- LipidFM v2: `01_continous_training/2_models/continued_pretrain/unimol2_lipids_84m_v2/checkpoints/`
  - `checkpoint_best.pt` — best valid loss (epoch 7) **← should be the one used**
  - `checkpoint_last.pt` — final epoch-8 weights
  - These are **Uni-Core format** (top-level keys: `args`, `model`, `loss`, `optimizer_history`, `task_state`, `extra_state`, `last_optimizer_state`). `torch.load(..., weights_only=False)` required.

### Reproducing the working frozen-feature benchmark
```
conda activate unimol2
python 01_continous_training/4_scripts/feature_benchmark.py --dataset all
```
Output: `5_outputs/finetune_moltrain/feature_benchmark_<ts>.json`

---

## 4. The Official Tutorial That Works (reference)

File: `code/document/unimol_QSRA_regression.ipynb`

```python
from unimol import MolTrain   # note: this notebook may use the OLD `unimol` package, not `unimol_tools`
clf = MolTrain(task='regression',
               data_type='molecule',
               epochs=50,
               learning_rate=0.0001,
               batch_size=16,
               early_stopping=10,
               metrics='mse',          # <-- they use 'mse', we use 'r2'
               split='random',
               save_path='./exp_reg_hERG_0616',
              )
clf.fit('datasets/hERG_train.csv')

predm = MolPredict(load_model='./exp_reg_hERG_0616')
pred_test_y = predm.predict('datasets/hERG_test.csv').reshape(-1)
mse = mean_squared_error(test_y, pred_test_y)   # evaluated on a SEPARATE test CSV
```

Key differences vs. our script:
1. **Official uses `metrics='mse'`; we use `metrics='r2'`.** This affects early-stopping/checkpoint-selection logic inside MolTrain.
2. **Official does a single 80/20 random split and evaluates with `MolPredict` on a separate held-out test CSV.** We set `kfold=5` and read `clf.cv_pred` (MolTrain's internal cross-validation predictions).
3. Official trains on **drug-like hERG molecules** (same distribution as Uni-Mol pretraining); we train on **flexible lipid molecules** (far from Uni-Mol pretraining distribution).

We have NOT yet tested whether switching to the official pattern (single split + `MolPredict` on held-out test + `metrics='mse'`) fixes the R².

---

## 5. Our Current Script (`finetune_moltrain.py`)

Path: `01_continous_training/4_scripts/finetune_moltrain.py`

Current key behavior:
- Loads checkpoint via monkey-patching `UniMolV2Model.load_pretrained_weights` to load our LipidFM checkpoint (`weights_only=False`). Logged: `[LipidFM] loaded 502 backbone keys into UniMolV2Model (head skipped: 2, unexpected: 6)` — checkpoint IS being loaded.
- Calls `MolTrain(..., metrics='r2', split='random', kfold=5, target_normalize='standard')`.
- After `clf.fit(csv)`, reads `clf.cv_pred` and `clf.data['target']`, inverse-scales targets, and computes `r2_score` / RMSE / Spearman.

### Known subtleties we discovered (verify these before/while fixing)
1. **Double de-normalization bug (already fixed):** `MolTrain.fit()` internally inverse-transforms `cv_pred` back to the modeling scale (`unimol_tools/train.py:282-288`), but `clf.data['target']` remains the *standardized* target. Our fix inverse-transforms `data['target']` with the same scaler so `y_true` and `y_pred` are on the same scale. Do NOT inverse-transform `cv_pred` again.
2. **Sheet3 must be `log10`'d** before feeding to MolTrain (we added this in `load_data`).
3. **JSON serialization crash:** `vars(args)` contains a `Path`; we now cast `Path` → `str` before `json.dumps`.
4. **`metrics='r2'` vs `metrics='mse'`:** unverified whether this is responsible for the breakdown. Test switching to `'mse'` as a single-fold sanity check (cheap).

---

## 6. Hypotheses to test (next agent)

We have several competing hypotheses that have NOT yet been conclusively isolated. Test them in this order (cheapest first).

### H1 — The `metrics='r2'` choice breaks MolTrain regression (MOST LIKELY, cheapest to test)
- MolTrain's trainer uses the `metrics` value to drive early-stopping and model selection. For regression with `target_normalize='standard'`, using `'r2'` may select/save a model based on an ill-behaved R² and/or break the head's convergence. 
- **Test:** rerun single-fold Sheet2 with `metrics='mse'` (match the official tutorial). If R² becomes healthy, this is the fix.

### H2 — `kfold` vs single split / `MolPredict` evaluation (also cheap to test)
- Official pattern: single 80/20 random split + `MolPredict` on a separate test CSV.
- Our pattern: `kfold=5` + read `clf.cv_pred` (internal CV).
- **Test:** change script to do a single `train_test_split`, fit on train CSV, and evaluate with `MolPredict` on the held-out test CSV (mimic the tutorial's Cells 31–32).

### H3 — Batch size too small (bsize=16 with 768–1920 samples)
- Maybe non-issue, but test `batch_size` up/down as a control.

### H4 — Lipid distribution vs Uni-Mol pretraining (likely a secondary factor, not the primary cause)
- Lipids are large/flexible and far from Uni-Mol's drug-like pretraining distribution. Even so, frozen-feature R² is good, so the backbone is usable; only end-to-end full-finetune under a too-large LR collapses. If H1/H2 don't fully fix it, consider:
  - lower LR (we tried 1e-4 → 5e-5, didn't help by itself),
  - `freeze_layers` to freeze most of the backbone and only train the head + last layers.

---

## 6. Environment / Commands

- Conda env: `unimol2` (Python 3.10; torch 2.13.0+cu130; RDKit 2026.03.5; `unimol_tools` 0.1.6; `unicore`).
- To run end-to-end (5-fold, lr=5e-5, checkpoint_best):
```
conda run -n unimol2 python 01_continous_training/4_scripts/finetune_moltrain.py --dataset all --epochs 20 --lr 5e-5 --kfold 5
```
- Single-fold sanity (cheap, ~1 hr):
```
conda run -n unimol2 python 01_continous_training/4_scripts/finetune_moltrain.py --dataset Sheet2 --epochs 20 --lr 5e-5 --kfold 1
```

> TIP: Run the single-fold Sheet2 first to iterate fast. Do NOT launch the full 4-hour `--dataset all` until the single-fold R² is healthy.

---

## 6. Deliverables expected from the next agent
1. Diagnose which hypothesis is the cause (start with H1: `metrics='mse'`).
2. Fix `finetune_moltrain.py` so end-to-end MolTrain fine-tuning reproduces healthy R² (≈ matching the frozen-feature results of ~0.57–0.62, or at least positive and stable).
3. Confirm both LipidFM v2 and stock-84M baseline both produce sane metrics under the same pipeline, so a valid baseline-vs-LipidFM comparison is possible.
4. Update `README.md` with the corrected workflow if needed.

## 7. Files touched / created this session
- `01_continous_training/4_scripts/finetune_moltrain.py` — modified (metrics, log10, de-norm, JSON Path).
- `01_continous_training/4_scripts/feature_benchmark.py` — created (frozen-feature baseline; healthy).
- `my_code/03_unimol2_finetune.py` — modified to point at `checkpoint_best.pt` (has a `weights_only` loading bug to fix for Uni-Core format).
- `my_code/utils.py` — minor cleanup.
- `code/README.md` — project overview.
- Outputs: `01_continous_training/5_outputs/finetune_moltrain/{feature_benchmark_<ts>.json, moltrain_results_<ts>.json}`.