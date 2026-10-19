# LipidFM — Ionizable-Lipid Foundation Model (Continued Pretraining)

Domain-adaptive continued pretraining of [Uni-Mol2](https://github.com/dptech-corp/Uni-Mol)
on ionizable lipids (ILs), producing a 3D molecular backbone for downstream LNP
property prediction (transfection efficiency, pKa, etc.).

Migrated from the EnviroFM environmental-toxicity pipeline. See
[Migration notes](#migration-notes-from-envirofm) for what changed and why —
several source defaults silently destroy this corpus.

## Current Development State

Please read the handoff document before continuing development:

```
/home/zx/df/disk1/my_project/lnpModel/prompt/26-08-07/03-交接.md
```

- Phase 0 (data merge): completed
- Phase 1 (virtual library): completed
- Phase 2 (continued pretraining): **completed** (84M, 6k steps, benchmark validated)
- **Next: Phase 3 (downstream finetuning: LNP transfection efficiency prediction)**
- Phase 2 benchmark: 5_outputs/benchmark_results.json

---

## Data layout

```
1_data/
├── raw/
│   ├── ionizable_lipids/pretrain_corpus.parquet   # Stage-1 output: 112,381 molecules
│   └── CHEMBLdb/ -> symlink                       # anti-forgetting replay source
└── processed/
    └── ionizable_lipids/unimolv2_3d/              # conformers + LMDB shards

2_models/
├── pretrained/
│   ├── checkpoint_84M.pt                          # Uni-Mol2 84M base
│   └── checkpoint_1.1B.pt                         # Uni-Mol2 1.1B base
└── continued_pretrain/                            # our adapted checkpoints

4_scripts/
└── Uni-Mol/                                      # vendored Uni-Mol2 task code (MIT)
```

The corpus is `real` (12,381 literature lipids) + `virtual` (100,000 enumerated),
a **1:8.08** ratio. See `../00_prepare_ILs_vitual_library/output/qc_report.md`.

---

## Pipeline

All parameters live in `3_configs/pipeline.yaml`.

### Step 1 — Prepare ionizable lipid 3D dataset

ETKDG + MMFF94s conformers, with ChEMBL replay molecules interleaved.

```bash
# Full dataset
python main.py --step prepare

# Smoke test (60 lipids + 15 ChEMBL)
python main.py --step prepare \
    --set prepare.sample_size=60 prepare.chembl_max=15 \
          prepare.output_dir=1_data/processed/smoke

# Resume an interrupted run
python main.py --step prepare --set prepare.resume=true
```

Outputs to `1_data/processed/ionizable_lipids/unimolv2_3d/`:
- `lipids_unimolv2_3d.sdf` — conformers, one SDF record each
- `lipids_unimolv2_failures.csv` — molecules that could not be processed
- `lipids_unimolv2_summary.json` — run statistics incl. `conformers_by_source`
- `lipids_unimolv2_checkpoint.json` — resume state

### Step 2 — Continued pretraining

The LMDB is built automatically on first run.

```bash
# Full run
python main.py --step pretrain

# Dry-run: print the unicore-train command without launching
python main.py --step pretrain --set pretrain.dry_run=true

# Rebuild LMDB from a new SDF, then train
python main.py --step pretrain --set pretrain.prepare_valid_split=true

# Train the 1.1B model instead
python main.py --step pretrain \
    --set pretrain.model_size=1.1b \
          pretrain.pretrained_checkpoint=2_models/pretrained/checkpoint_1.1B.pt \
          pretrain.output_dir=2_models/continued_pretrain/unimol2_lipids_1.1b
```

Checkpoints land in `<output_dir>/checkpoints/`.

### Run all steps

```bash
python main.py
```

---

## Migration notes (from EnviroFM)

Ionizable lipids have a **median of 39 rotatable bonds** vs ~5 for drug-like
molecules. Three source defaults were measured to fail on this corpus, all
silently — they discard molecules rather than erroring:

| Parameter | Source | Here | Measured effect |
|---|---|---|---|
| `useRandomCoords` | unset | **`true`** | Molecules reaching 5 conformers (60–140 heavy atoms): **8.3% → 100%** |
| `max_iterations` | 500 | **2000** | MMFF94s convergence: **0% → 95.8%** |
| `max_atoms` | 128 | **256** | 128 would silently crop **2.5%** of the corpus (max is 208 heavy atoms) |

Verified end-to-end: the unmodified source parameters succeeded on 29% of a
stratified sample (0% above 60 heavy atoms); the adapted settings succeed on
40/40.

### Other adaptations

| Aspect | Source | Here | Why |
|---|---|---|---|
| Input format | CSV (chunked) | **parquet** | Stage-1 output; carries `source_type` |
| ID / SMILES column | `DTXSID` / `SMILES` | `inchikey` / `smiles_std` | — |
| `qsar_col` | `QSAR_READY_SMILES` | *removed* | Not present in our corpus |
| Conformer count | 10→5 for all | **real 24→20, virtual 6→4** | Real lipids are scarce; see note below |
| `learning_rate` | 1e-4 | **5e-5** | Corpus ~10× smaller → lower forgetting risk |
| Benchmark step | `run_benchmark.py` | *not migrated* | No downstream IL benchmark yet (Stage 3) |
| Conformer gate | must reach `keep_conformers` | keep whatever converged | A strict gate discards usable flexible-lipid structures |

### Conformer counts do not change sample counts

`ConformerSampleDataset` draws **one conformer per molecule per epoch**
(`unimol2/data/conformer_sample_dataset.py:34`):

```python
sample_idx = np.random.randint(size)
```

So conformer count is **3D data augmentation**, not replication — the effective
real:virtual sample ratio is set purely by molecule counts (1:8.08). Real lipids
get 20 conformers because they are scarce and worth covering densely, not to
rebalance the mix.

### Do not raise `prepare.max_atoms` above `pretrain.max_atoms`

`CroppingDataset` (`unimol2/data/cropping_dataset.py:36`) responds to oversized
molecules by *randomly sampling atoms*:

```python
index = np.random.choice(len(atoms), self.max_atoms, replace=False)
```

This produces a chemically meaningless atom cloud with no error raised.
`main.py` refuses to run if the two values are inconsistent.

---

## Environment

**Two different interpreters are required** — this is not optional:

| Step | Interpreter | Why |
|---|---|---|
| Step 1 (prepare) | `conda activate unimol2` | RDKit conformer generation |
| Step 2 (pretrain) | `.../Uni-Mol2_Toxicity_Prediction/.venv/bin/python` | **only this env has `unicore`** |

The `unimol2` conda env has no `unicore` package, so `unicore-train` is missing
and Step 2 cannot launch from it. Use the source project's `uv` venv:

```bash
VENV=/home/zx/df/disk1/my_project/Uni-Mol2_Toxicity_Prediction/.venv
export PATH="$VENV/bin:$PATH"
$VENV/bin/python main.py --step pretrain
```

Verified working: torch 2.6.0+cu124, unicore 0.0.1, RDKit 2026.03.3, CUDA available.

---

## Verification status

Smoke-tested end to end (40 lipids + 10 ChEMBL, 20 training steps):

- Conformer generation: **72/72 molecules succeeded, 0 failures**
- SDF alignment contract (`atoms == AddHs(MolFromSmiles(SMILES))`): **176/176**
- LMDB records: correct `{smi, atoms, coordinates}` structure, 36/36 aligned
- Checkpoint load: **502/502 keys, 0 missing, 0 unexpected**
- Training: loss 0.493 → 0.418, valid 0.229, `masked_acc` 98.9%, 481 tensors updated

---

## Project structure

```
01_continous_training/
├── main.py                                  # Pipeline entry point
├── 3_configs/pipeline.yaml                  # All parameters
├── 4_scripts/
│   ├── prepare_lipids_unimol_dataset.py     # Step 1: 3D conformers + ChEMBL replay
│   ├── build_lipids_pretrain_lmdb.py        # Step 2a: SDF → LMDB shards
│   ├── run_lipids_continue_pretrain.py      # Step 2b: continued pretraining
│   └── Uni-Mol/                             # symlink to vendored Uni-Mol/Uni-Core
├── 1_data/ 2_models/ 5_outputs/
└── logs/ error/
```

## Hardware

Single RTX 4090 (48 GB). `batch_size × update_freq = 8 × 4 = 32` effective batch
for the 84M model; the 1.1B model will need a smaller `batch_size` and a larger
`update_freq` to fit.
