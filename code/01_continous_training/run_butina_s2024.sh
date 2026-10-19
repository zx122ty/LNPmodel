#!/usr/bin/env bash
# Re-run the s2024 Butina partition. The first attempt died mid-fold-3 with
# "CUDA error: unknown error" and the device then dropped off the bus entirely
# (nvidia-smi: "No devices were found"). That was hardware, not code: folds 0-2 and
# every other run on this machine had completed normally.
#
# MolTrain has no per-fold resume -- cv.data is written once at the end of fit() --
# so a 3/5-complete run has to restart from fold 0. The partial artifacts were
# removed rather than reused, so nothing from the crashed run can contaminate this one.
set -euo pipefail
cd "$(dirname "$0")"
V5CKPT="$PWD/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
  4_scripts/finetune_moltrain.py \
    --dataset Sheet2 --split butina --butina-cutoff 0.7 --seed 2024 \
    --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 \
  2>&1 | tee "$PWD/logs/finetune_v5_butina_2d_s2024.log"
