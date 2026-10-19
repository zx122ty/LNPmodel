#!/usr/bin/env bash
# run_v6b_da.sh —— LipidFM v6b 预训练（v5 严格对照：仅改描述符 32→215）
#
# v6b = v5 语料/构象/损失/head/训练 完全一致，唯一变量 = MTR 描述符集：
#   v5 : 32 curated 描述符
#   v6b: 全量 RDKit (~212) + 3 脂质先验 = 215（v6_descriptors.py）
# 数据直接复用 v6 LMDB（215 列 + z-stats + epoch symlinks 已就绪）。
#
# 归因目标：描述符增强是否独立有效（排除 v6 的 MTR-only/head/pooling 变量）。
#
# 用法：
#   bash run_v6b_da.sh            # 30k 步
set -euo pipefail

CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
STEPS="${1:-30000}"
CONFIG="$CT/3_configs/pipeline_v6b.yaml"
OUT_DIR="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6b"
LOG_DIR="$CT/logs"
LOG="$LOG_DIR/pretrain_v6b_${STEPS}.log"
LMDB="$CT/1_data/processed/ionizable_lipids/unimolv2_3d_v6/lmdb_pretrain/lmdb_manifest.json"

cd "$CT"
mkdir -p "$LOG_DIR"

if [[ ! -f "$LMDB" ]]; then
  echo "缺少 v6 LMDB manifest：$LMDB（v6b 复用 v6 数据）"
  exit 1
fi

CKPT="$OUT_DIR/checkpoints/checkpoint_last.pt"
if [[ -f "$CKPT" ]] && grep -q "num_updates: $STEPS >= max_update: $STEPS" "$LOG" 2>/dev/null; then
  echo "已完成 $STEPS 步，跳过。checkpoint：$CKPT"
  exit 0
fi

echo "=== LipidFM v6b 预训练（仅描述符 32→215，30k 步）==="
echo "LMDB：$LMDB（215 列）"
echo "配置：$CONFIG"
echo "输出：$OUT_DIR"
echo "日志：$LOG"

conda run -n unimol2 --no-capture-output python main.py --step pretrain \
  --config "$CONFIG" \
  --set pretrain.total_steps="$STEPS" \
  2>&1 | tee "$LOG"

echo "=== 完成。下一步：评估（frozen 快筛 + finetune 确认）==="
echo "  bash run_v6b_eval_queue.sh"
