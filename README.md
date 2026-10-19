# LipidFM

LipidFM is a Uni-Mol2-based foundation model for **ionizable lipids** and
lipid nanoparticle (LNP) property prediction. This repository provides the
deployment inference pipeline, the pretraining and fine-tuning code, chemically
aware evaluation splits, and a lightweight CPU benchmark.

The current production-oriented predictor accepts lipid SMILES and returns:

- predicted transfection efficiency on the `log2(RLU)` scale;
- a calibrated uncertainty estimate;
- the equivalent prediction and one-sigma bounds on the raw RLU scale;
- an optional out-of-domain similarity check against the training library.

For scientific reproduction details, metric definitions, known pitfalls, and
the full training pipeline, see [code/README.md](code/README.md).

## Model overview

The validated inference ensemble combines three predictors:

```text
mu_log2 = 0.564 * neural_2d
        + 0.130 * traditional
        + 0.306 * atom_pair
```

where:

- `neural_2d` is a fine-tuned LipidFM / Uni-Mol2 fold checkpoint using molecular
  graph and 3D conformer inputs;
- `traditional` is an XGBoost model over frozen RDKit descriptors plus Morgan
  fingerprints;
- `atom_pair` is an XGBoost model over atom-pair fingerprints;
- uncertainty is predicted by a random-forest model and multiplied by a fixed
  calibration factor of `1.9`.

The blend weights are frozen from the validated scaffold protocol and are not
refit at inference time. The bundled `manifest.json` is the source of truth for
the weights and feature definitions.

The deployment assets in this release use the LipidFM v5 checkpoint:
`unimol2_lipids_84m_v5/checkpoint_1_30000.pt`, fine-tuned for 40 epochs on the
Sheet2 scaffold/seed-42 fold. The neural component is loaded from the complete
fold model `model_0.pth`; the continued-pretraining backbone checkpoint is not
required at inference time.

## Scope and limitations

LipidFM is intended for ionizable-lipid screening and prioritization. It is not
a replacement for synthesis or biological experiments.

- The deployment model was validated on a 4CR ionizable-lipid library.
  Structurally unrelated molecules are extrapolation even when the output looks
  numerically plausible.
- Use `--report-ood` to compute the maximum ECFP4 Tanimoto similarity to the
  training set. Inspect this value before trusting predictions for a new chemical
  series.
- Rank candidates by `mu_log2_RLU`. The uncertainty estimate is useful for
  assessing trust, but it is not the validated ranking signal.
- `mu ± sigma` is a calibrated error bar, not a strict 90% conformal interval.
- Predictions use the `log2(RLU)` target scale from Sheet2. Do not compare them
  directly with the Sheet3 benchmark model, which uses `log10` targets.

## Repository layout

```text
.
├── benchmark/                              # Lightweight CPU benchmark and results
├── code/
│   ├── 00_prepare_ILs_vitual_library/       # Corpus construction
│   ├── 00_prepare_ILs_vitual_library_v2/    # PubChem/Enamine Ugi-4CR libraries
│   └── 01_continous_training/
│       ├── 4_scripts/predict_lipids.py      # Main inference entry point
│       ├── 4_scripts/Uni-Mol/               # Vendored Uni-Mol2 task code (MIT)
│       └── 5_outputs/                       # Downloaded deployment assets
├── tools/download_hf_assets.py              # Hugging Face asset downloader
├── ASSETS.md                                # Required Hugging Face layout
├── requirements.txt
└── install_env.sh
```

Large datasets, checkpoints, and fine-tuned weights are excluded from Git and
published separately on Hugging Face.

## Requirements

Recommended:

- Linux;
- Python 3.10;
- an NVIDIA GPU with a CUDA-compatible PyTorch installation for the neural arm;
- approximately 1 GB of free disk space for the deployment assets and runtime
  caches.

The neural arm can run on CPU, but inference is slower. The `--no-neural` mode is
CPU-only and intentionally unsupported for production conclusions.

## Installation

### 1. Clone the repository

```bash
git clone <GITHUB_REPOSITORY_URL>
cd lnpModel
```

### 2. Create the environment

For inference only, a regular virtual environment is sufficient:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build that matches the host CUDA driver when GPU inference is
required. Uni-Core is not imported by the deployment predictor.

For training, or to reproduce the environment used on the reference machine, use
the Conda installer below.

The repository includes the Uni-Core and Uni-Mol2 task source required by the
training and inference code:

```bash
bash install_env.sh
conda activate unimol2
```

To install the environment at a custom prefix:

```bash
CONDA_ENV_PREFIX=/opt/conda/envs/unimol2 bash install_env.sh
conda activate /opt/conda/envs/unimol2
```

Verify the runtime:

```bash
python -c "import torch, rdkit, unicore, unimol_tools; print('environment OK')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If you only need the lightweight benchmark model under `benchmark/code/`, it has
its own smaller dependency list:

```bash
cd benchmark/code
python -m pip install -r requirements.txt
```

### 3. Download the deployment assets

Hugging Face should mirror the repository directory structure. Set `HF_REPO_ID`
to the asset repository published with this release, for example
`your-org/LipidFM-assets`:

```bash
python tools/download_hf_assets.py --repo-id <HF_REPO_ID>
```

For a private repository:

```bash
huggingface-cli login
python tools/download_hf_assets.py --repo-id <HF_REPO_ID>
```

For a Hugging Face Dataset repository, add `--repo-type dataset`.

The minimum deployment bundle is:

```text
code/01_continous_training/5_outputs/deploy_models/
├── manifest.json
├── desc_spec.npz
├── traditional.json
├── xgb_atompair.json
└── sigma_hat_rf.joblib

code/01_continous_training/5_outputs/finetune_moltrain/
└── Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed42_lr5e-05_ep40/
    ├── model_0.pth
    ├── target_scaler.ss
    └── config.yaml
```

`benchmark/input_data/dataset.xlsx` is additionally required only when using
`--report-ood`. See [ASSETS.md](ASSETS.md) for the complete asset manifest.

## Run inference

### Single SMILES

```bash
cd code/01_continous_training

python 4_scripts/predict_lipids.py \
  --smiles "CCCCCCCC(=O)N(CCN(C)C)C(CCCCC)C(=O)NC" \
  --device cuda
```

Set `--device cpu` when CUDA is unavailable. The script also falls back to CPU if
CUDA is not detected.

### Batch CSV

The input file must contain one SMILES column:

```csv
id,SMILES
candidate_1,CCCCCCCC(=O)N(CCN(C)C)C(CCCCC)C(=O)NC
candidate_2,CCCCOC(=O)C(C)SCC(COC(=O)CCC)OC(=O)CCOC(=O)CCNCCO
```

```bash
python 4_scripts/predict_lipids.py \
  --input candidates.csv \
  --smiles-col SMILES \
  --top 100 \
  --report-ood \
  --out ranked.csv \
  --device cuda
```

### CPU plumbing check

```bash
python 4_scripts/predict_lipids.py \
  --smiles "CCCCCCCC(=O)N(CCN(C)C)C(CCCCC)C(=O)NC" \
  --no-neural
```

`--no-neural` mixes only the two CPU arms and renormalizes their weights. It is a
deployment and data-plumbing check, not the validated predictor.

## Output

The CLI prints the first 20 rows and optionally writes a CSV:

| Column | Description |
|---|---|
| `smiles` | Input SMILES that passed RDKit parsing |
| `mu_log2_RLU` | Predicted transfection efficiency in `log2(RLU)` |
| `sigma_calibrated_log2` | Calibrated uncertainty in log2 units |
| `mu_minus_1sigma_log2` | Lower one-sigma bound in log2 units |
| `mu_plus_1sigma_log2` | Upper one-sigma bound in log2 units |
| `predicted_RLU` | `2 ** mu_log2_RLU` |
| `lower_1sigma_RLU` | `2 ** (mu - sigma)` |
| `upper_1sigma_RLU` | `2 ** (mu + sigma)` |
| `max_tanimoto_to_train` | Optional maximum ECFP4 similarity to the training set |

Invalid SMILES are reported and excluded rather than silently replaced.

## Deployment notes

The supported deployment interface in this release is the CLI entry point:

```text
4_scripts/predict_lipids.py
```

For a service, run the CLI inside an isolated worker or container, preserve the
repository-relative asset layout, and return both the log2 prediction and the
uncertainty fields. Do not expose `--no-neural` as a production mode.

Recommended request validation:

1. Require a non-empty SMILES string.
2. Parse it with RDKit before accepting the request.
3. Reject malformed structures with a client error.
4. Run OOD similarity checks for new chemical series.
5. Store the GitHub release, Hugging Face revision, and asset checksums used for
   each production result.

The deployment scripts support custom asset locations through:

```bash
LIPIDFM_DEPLOY_MODELS_DIR=/path/to/deploy_models
LIPIDFM_FINETUNED_MODEL_DIR=/path/to/finetuned_fold
LNP_BENCHMARK_XLSX=/path/to/dataset.xlsx
```

## Lightweight CPU benchmark

`benchmark/code/` contains a separate RandomForest + split-conformal model for
the Sheet3 transfection dataset:

```bash
cd benchmark/code
python scripts/predict.py \
  --smiles "CCCCOC(=O)C(C)SCC(COC(=O)CCC)OC(=O)CCOC(=O)CCNCCO"
```

This model:

- is CPU-only;
- predicts on the `log10` scale;
- returns a 90% conformal interval;
- is not the LipidFM neural ensemble and should not be mixed with its metrics.

See [benchmark/code/README.md](benchmark/code/README.md) for details.

## Training and reproduction

The complete three-stage pipeline is documented in
[code/README.md](code/README.md):

1. construct the ionizable-lipid corpus and virtual library;
2. generate 3D conformers and continue-pretrain Uni-Mol2;
3. fine-tune and evaluate on transfection, pKa, and MTR tasks.

Common entry points:

```bash
cd code/01_continous_training

python main.py --step prepare
python main.py --step pretrain --set pretrain.dry_run=true
python main.py --step pretrain

python 4_scripts/finetune_moltrain.py \
  --dataset all --split scaffold --epochs 20 --lr 1e-4 --kfold 5
```

Training requires the additional data and backbone assets listed in
[ASSETS.md](ASSETS.md).

## Evaluation protocol

- Report `scaffold` splits for the main generalization evaluation.
- Use `butina` splits as a chemical-cluster robustness diagnostic.
- Treat `random` splits as interpolation-oriented baseline comparisons.
- Sheet3 uses `log10(RLU)` while the deployment ensemble uses `log2(RLU)`.
  RMSE values are therefore not directly comparable across those targets.
- Frozen scaffold, Butina, random, and sequential split files are included in
  the Hugging Face bundle under
  `code/01_continous_training/5_outputs/downstream_splits/`.

## Troubleshooting

### `CUDA available: False`

Install a PyTorch build compatible with the host driver, or run:

```bash
python 4_scripts/predict_lipids.py --smiles "<SMILES>" --device cpu
```

### Missing model or descriptor-spec error

The deployment assets were not restored. Rerun the Hugging Face downloader and
verify the paths listed above.

### Hugging Face downloads are slow

Set an accessible endpoint before downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python tools/download_hf_assets.py --repo-id <HF_REPO_ID>
```

### RDKit rejects an input

Invalid or unsupported SMILES are excluded from batch prediction. Validate and
standardize the structures before deployment.

## License

Uni-Core and the vendored Uni-Mol2 task source retain their upstream licenses;
see their respective `LICENSE` files and
`code/01_continous_training/4_scripts/Uni-Mol/VENDORED.md`. Add the project-level
license selected for the GitHub release before redistribution.
