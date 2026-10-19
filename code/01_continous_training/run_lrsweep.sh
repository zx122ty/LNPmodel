#!/usr/bin/env bash
# LR sweep: does v3 just need different fine-tuning hyperparameters?
#
# Both ends are tuned (stock AND v3) so the comparison is best-vs-best. Tuning only
# v3 would be rigged: stock's 0.639 was itself an untuned lr=1e-4 point.
#
# epochs 20 -> 40, early-stopping 5 -> 8: at lr=1e-4 several folds were still
# improving at epoch 20/20 (stock fold1/fold2 peaked AT the cap), so 20 epochs
# truncates training. Lower lrs converge slower still, and would be penalised by a
# cap that never binds for the high-lr arm.
set -u
cd /home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training
LOGD=logs/lrsweep_20260818
mkdir -p $LOGD
CKPT_DIR=$PWD/2_models/continued_pretrain
COMMON="--dataset Sheet2 --split scaffold --kfold 5 --seed 42 --epochs 40 --early-stopping 8"

run () {
  local name=$1 lr=$2; shift 2
  local log=$LOGD/${name}_lr${lr}.log
  if grep -qa "MolTrain\] fold-mean" $log 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP $name lr=$lr (already complete)"; return
  fi
  echo "[$(date '+%F %T')] START $name lr=$lr -> $log"
  env "$@" conda run -n unimol2 --no-capture-output \
    python 4_scripts/finetune_moltrain.py $COMMON --lr $lr > $log 2>&1
  echo "[$(date '+%F %T')] DONE  $name lr=$lr rc=$?"
}

# Interleave the arms: if the sweep is cut short we still have a usable
# stock-vs-v3 pair at each lr, instead of three tuned stock runs and no v3.
for lr in 3e-5 5e-5 1e-4; do
  run stock $lr USE_STOCK_84M=1
  run v3    $lr LIPIDFM_CHECKPOINT=$CKPT_DIR/unimol2_lipids_84m_v3/checkpoints/checkpoint_best.pt
done
echo "[$(date '+%F %T')] LR SWEEP COMPLETE"
