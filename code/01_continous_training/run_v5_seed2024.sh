#!/usr/bin/env bash
# Third repetition of the scaffold partition for the repeated-CV comparison.
# Runs v5 with the SAME protocol as the s1337 arm so the traditional-vs-neural
# comparison is paired on folds, not just on the dataset.
set -euo pipefail
cd "$(dirname "$0")"
V5CKPT="$PWD/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
  4_scripts/finetune_moltrain.py \
    --dataset Sheet2 --split scaffold --seed 2024 \
    --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 \
  2>&1 | tee "$PWD/logs/finetune_v5_Sheet2_scaf_s2024.log"
