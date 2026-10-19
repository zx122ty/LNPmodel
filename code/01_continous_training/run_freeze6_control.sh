#!/usr/bin/env bash
# 补齐「freeze6 无 adapter」在 seed42 / 2024 的对照。
#
# 为什么必须补：历史上 freeze6 只跑过 seed1337（单分区、无重复），
# 其报告值 0.6482 只存在于日志里——产物已被后来的 baseline 重跑覆盖（坑 #6）。
# 没有同折对照就无法判断「adapter 在 freeze6 下是否真的没贡献」，
# 而这恰恰是本次最重要的结论之一。用现有 --freeze-layers 机制即可，不需要 adapter。
set -uo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"

V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"

for seed in 42 2024; do
  LOG="logs/freeze6_noadapter_scaffold_s${seed}.log"
  echo "########## freeze6 (NO adapter) scaffold seed=$seed  $(date '+%F %T')"
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -lt 1000 ]; then break; fi
    sleep 60
  done
  env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py \
    --dataset Sheet2 --split scaffold --seed "$seed" \
    --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 \
    --freeze-layers 6 \
    > "$LOG" 2>&1 || echo "!!!! FAILED (see $LOG)"
  echo "########## done $(date '+%F %T')"
done
echo "########## FREEZE6 CONTROL COMPLETE $(date '+%F %T')"
