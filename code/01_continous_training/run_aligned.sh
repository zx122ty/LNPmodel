#!/usr/bin/env bash
# Aligned scaffold-split comparison: stock / v2 / v3 on byte-identical cached folds.
set -u
cd /home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training
LOGD=logs/aligned_20260817
CKPT_DIR=$PWD/2_models/continued_pretrain
COMMON="--split scaffold --lr 1e-4 --epochs 20 --kfold 5"

run () {  # name, dataset, seed, env-prefix...
  local name=$1 ds=$2 seed=$3; shift 3
  local log=$LOGD/${name}_${ds}_seed${seed}.log
  echo "[$(date '+%F %T')] START $name $ds seed=$seed -> $log"
  env "$@" conda run -n unimol2 --no-capture-output \
    python 4_scripts/finetune_moltrain.py --dataset $ds --seed $seed $COMMON \
    > $log 2>&1
  echo "[$(date '+%F %T')] DONE  $name $ds seed=$seed rc=$?"
}

# --- Sheet2: single seed (low variance, +-0.05) ---
run stock Sheet2 42 USE_STOCK_84M=1
run v2    Sheet2 42 LIPIDFM_CHECKPOINT=$CKPT_DIR/unimol2_lipids_84m_v2/checkpoints/checkpoint_best.pt
run v3    Sheet2 42 LIPIDFM_CHECKPOINT=$CKPT_DIR/unimol2_lipids_84m_v3/checkpoints/checkpoint_best.pt

# --- Sheet3: 3 seeds (n=768, per-fold std up to +-0.49) ---
for s in 42 1337 2024; do
  run stock Sheet3 $s USE_STOCK_84M=1
  run v2    Sheet3 $s LIPIDFM_CHECKPOINT=$CKPT_DIR/unimol2_lipids_84m_v2/checkpoints/checkpoint_best.pt
  run v3    Sheet3 $s LIPIDFM_CHECKPOINT=$CKPT_DIR/unimol2_lipids_84m_v3/checkpoints/checkpoint_best.pt
done
echo "[$(date '+%F %T')] ALL RUNS COMPLETE"
