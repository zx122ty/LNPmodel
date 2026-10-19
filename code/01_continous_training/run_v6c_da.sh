#!/usr/bin/env bash
# run_v6c_da.sh —— LipidFM v6c 预训练（v5 严格对照：仅改 MTR head 论文式）
#
# v6c = v5 语料/构象/损失/描述符/训练 完全一致，唯一变量 = MTR head：
#   v5 : 1层MLP + tanh + CLS（hidden 256）
#   v6c: 2层MLP + ReLU + Dropout0.1 + mean pooling（hidden 512，论文式）
# 数据用 v5 LMDB（32 描述符，与 v5 完全一致）。
#
# 归因目标：MTR head 独立效应（v6 三变量析因的最后一块：v6b=描述符、v6c=head，
# 余下 MTR-only 效应 = v6 − v6b − v6c 联合差可解）。
#
# 用法：
#   bash run_v6c_da.sh            # 30k 步
set -euo pipefail

CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
STEPS="${1:-30000}"
CONFIG="$CT/3_configs/pipeline_v6c.yaml"
OUT_DIR="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6c"
LOG_DIR="$CT/logs"
LOG="$LOG_DIR/pretrain_v6c_${STEPS}.log"
LMDB="$CT/1_data/processed/ionizable_lipids/unimolv2_3d_v5/lmdb_pretrain/lmdb_manifest.json"

cd "$CT"
mkdir -p "$LOG_DIR"

if [[ ! -f "$LMDB" ]]; then
  echo "缺少 v5 LMDB manifest：$LMDB"
  exit 1
fi

CKPT="$OUT_DIR/checkpoints/checkpoint_last.pt"
if [[ -f "$CKPT" ]] && grep -q "num_updates: $STEPS >= max_update: $STEPS" "$LOG" 2>/dev/null; then
  echo "已完成 $STEPS 步，跳过。checkpoint：$CKPT"
  exit 0
fi

echo "=== LipidFM v6c 预训练（仅 MTR head 论文式，30k 步）==="
echo "LMDB：$LMDB（v5，32 描述符）"
echo "配置：$CONFIG"
echo "输出：$OUT_DIR"
echo "日志：$LOG"

conda run -n unimol2 --no-capture-output python main.py --step pretrain \
  --config "$CONFIG" \
  --set pretrain.total_steps="$STEPS" \
  2>&1 | tee "$LOG"

echo "=== 完成。下一步：评估（frozen 快筛 + finetune 确认）==="
echo "  bash run_v6c_eval_queue.sh"
