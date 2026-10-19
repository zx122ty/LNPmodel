#!/usr/bin/env bash
# 冻结层数扫描：找到 partial-freeze 的甜点。
#
# 已有的两个端点（Sheet2 scaffold）：
#   --freeze-layers 0  = 全参微调 A        0.6708 (seed42)
#   --freeze-layers 6  = D0（本次正向发现） 0.6584 (seed42)
#   --freeze-layers 12 走的是 --freeze-backbone 那套前缀，不等价，不在本扫描内。
#
# 本脚本补 4 / 8 / 10 三个点，先在 seed42 上扫，出曲线后再把胜者扩到 1337/2024。
#
# ⚠️ 注意 --freeze-layers 只冻 encoder.layers.*，embed_tokens / atom_feature /
#    movement_pred_head 等仍然可训（约 2.5M 参数）。所以 freeze12 ≠ 配置 B。
set -uo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"

V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
SEED="${1:-42}"
LAYERS="${2:-4 8 10}"

for n in $LAYERS; do
  LOG="logs/freeze_scan_f${n}_scaffold_s${SEED}.log"
  echo "########## freeze-layers=$n scaffold seed=$SEED  $(date '+%F %T')"
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -lt 1000 ]; then break; fi
    sleep 60
  done
  env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py \
    --dataset Sheet2 --split scaffold --seed "$SEED" \
    --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 \
    --freeze-layers "$n" \
    > "$LOG" 2>&1 || echo "!!!! FAILED (see $LOG)"
  echo "########## done f$n  $(date '+%F %T')"
done
echo "########## SCAN COMPLETE $(date '+%F %T')"
