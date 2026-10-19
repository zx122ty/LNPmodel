#!/usr/bin/env bash
# Complete the 3D-conformer arm on the remaining two scaffold partitions so the
# 2D+3D ensemble claim can be tested on 15 folds instead of 5.
set -euo pipefail
cd "$(dirname "$0")"
V5CKPT="$PWD/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
for S in 42 2024; do
  echo "=== 3D arm seed $S ==="
  env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py \
      --dataset Sheet2 --split scaffold --seed "$S" \
      --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 \
      --use-3d-conformers \
    2>&1 | tee "$PWD/logs/finetune_v5_3dconf_Sheet2_scaf_s${S}.log"
done
