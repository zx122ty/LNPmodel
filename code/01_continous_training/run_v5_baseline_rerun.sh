#!/usr/bin/env bash
# Baseline half of the paired conformer test: v5 with MolTrain's BUILT-IN conformer
# generator (the path that silently substitutes 2D for ~82% of these molecules).
# Re-run now so both arms share code, environment and day -- the historical 0.673
# artifact had its model_0.pth overwritten by an aborted 3D run, so its scaler and
# cv.data can no longer be assumed to belong to the same training pass.
set -euo pipefail
cd "$(dirname "$0")"
V5CKPT="$PWD/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
  4_scripts/finetune_moltrain.py \
    --dataset Sheet2 --split scaffold --seed 1337 \
    --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 \
  2>&1 | tee "$PWD/logs/finetune_v5_baseline_rerun_Sheet2_scaf_s1337.log"
