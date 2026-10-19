#!/usr/bin/env bash
# Confirm the neural-vs-traditional REVERSAL on Butina across more partitions.
#
# One Butina partition (seed 42) showed traditional beating neural_2d (0.7887 vs
# 0.7652, 4/5 folds, p=0.050) -- the OPPOSITE of the scaffold result (neural +0.036,
# p=0.0035, 14/15 folds). That reversal is the most consequential finding so far, but
# 5 folds from a single partition is thin evidence. This adds two more partitions so
# the Butina side matches the scaffold side's 15-fold power.
#
# Only the neural_2d arm is needed: the reversal under test is neural_2d vs the
# traditional model, and the traditional model is instant on CPU (already computed).
set -euo pipefail
cd "$(dirname "$0")"
V5CKPT="$PWD/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
for S in 1337 2024; do
  # Skip partitions that already have trained folds, so re-running the script
  # after a partial failure cannot clobber a completed run's checkpoints.
  if compgen -G "5_outputs/finetune_moltrain/*butina_seed${S}_lr*/*.pth" > /dev/null; then
    echo "=== butina v5 2d seed $S: already complete, skipping ==="
    continue
  fi
  echo "=== butina v5 2d seed $S ==="
  env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py \
      --dataset Sheet2 --split butina --butina-cutoff 0.7 --seed "$S" \
      --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 \
    2>&1 | tee -a "$PWD/logs/finetune_v5_butina_2d_s${S}.log"
done
