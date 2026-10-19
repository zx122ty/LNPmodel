#!/usr/bin/env bash
# PEFT 消融（方向 7 / P0-2）：冻结主干 + 瓶颈 Adapter vs 全参微调
#
# A/B/C/D 矩阵（方案 §5.3），本脚本按参数选择要跑哪一档：
#   A  全参微调（基线，已存在，无需重跑）      scaffold 15 折 0.6807 / butina 0.7654
#   B  --frozen                               仅线性头可训（769 参数）
#   C  --frozen --adapter 48                  仅 adapter+头可训（913,729 参数 = 1.083%）
#   D  --freeze6  --adapter 48                冻结前 6 层（沿用已有 freeze-layers 语义）
#
# 用法：
#   bash run_peft_eval.sh B scaffold 42      # 冻结主干 + 线性头
#   bash run_peft_eval.sh C scaffold 42      # 冻结主干 + adapter
#   bash run_peft_eval.sh D scaffold 42      # 冻结前6层 + adapter
#
# 注意：
#   * MolTrain 无折级断点续训（坑 #8）——中断需从 fold 0 重跑，故一次只排一个配置。
#   * GPU 已掉线 3 次；本脚本等 GPU 空闲再启动，但不会自动重试。
set -euo pipefail

CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"

# NB: no '}' inside the ${n:?word} messages — an unbalanced brace there closes the
# parameter expansion early and the rest of the message leaks into the variable.
USAGE="usage: run_peft_eval.sh B|C|D scaffold|butina [seed]"
CONFIG="${1:?$USAGE}"
SPLIT="${2:?$USAGE}"
SEED="${3:-42}"

V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
ADAPTER_DIM=48

case "$CONFIG" in
  B) PEFT_FLAGS="--freeze-backbone" ;;
  C) PEFT_FLAGS="--freeze-backbone --adapter-dim $ADAPTER_DIM" ;;
  D) PEFT_FLAGS="--freeze-layers 6 --adapter-dim $ADAPTER_DIM" ;;
  *) echo "unknown config: $CONFIG (expected B, C or D)"; exit 2 ;;
esac

# butina 必须显式 0.7（坑：命令行默认 0.6 会产生不均衡划分 [332,333,332,332,591]）
BUTINA_FLAGS=""
if [ "$SPLIT" = "butina" ]; then BUTINA_FLAGS="--butina-cutoff 0.7"; fi

LOG="logs/peft_${CONFIG}_Sheet2_${SPLIT}_s${SEED}.log"

echo "===== waiting for GPU to be free $(date '+%F %T')"
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-0}" -lt 1000 ]; then echo "GPU free ($used MiB)"; break; fi
  sleep 60
done

echo "===== [peft_${CONFIG}] split=${SPLIT} seed=${SEED} $(date '+%F %T')"
env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
  4_scripts/finetune_moltrain.py \
  --dataset Sheet2 --split "$SPLIT" $BUTINA_FLAGS --seed "$SEED" \
  --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 \
  $PEFT_FLAGS \
  > "$LOG" 2>&1 || echo "!!!! FAILED (see $LOG)"
echo "===== done $(date '+%F %T')"
