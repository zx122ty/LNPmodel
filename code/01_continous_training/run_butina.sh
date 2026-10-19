#!/usr/bin/env bash
# Butina-cluster CV: external-validity check on a DIFFERENT partition criterion.
# All prior 15 folds share scaffold partitions, so they are not fully independent
# repetitions. Butina tests generalisation across chemical space (Tanimoto) rather
# than across topology (Murcko scaffolds).
#
# Cutoff note: 0.4 (a common default) is unusable on this dataset -- the 4CR library
# is a full combinatorial enumeration, so at 0.4 Tanimoto 1914/1920 molecules collapse
# into ONE cluster. 0.7 gives 150 clusters and balanced [384,384,384,384,384] folds.
set -euo pipefail
cd "$(dirname "$0")"
V5CKPT="$PWD/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
for ARM in "" "--use-3d-conformers"; do
  TAG=$([ -z "$ARM" ] && echo "2d" || echo "3dconf")
  echo "=== butina v5 $TAG ==="
  env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py \
      --dataset Sheet2 --split butina --butina-cutoff 0.7 --seed 42 \
      --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 $ARM \
    2>&1 | tee "$PWD/logs/finetune_v5_butina_${TAG}.log"
done
