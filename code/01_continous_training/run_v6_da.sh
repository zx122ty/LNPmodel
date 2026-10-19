#!/usr/bin/env bash
# run_v6_da.sh —— LipidFM v6 论文式 MTR-only 领域适应（DA）
#
# v6 = v5 语料（300k IL + 700k ChEMBL，构象复用）+ 论文式 MTR：
#   * MTR 目标：全量 RDKit 描述符（~200）+ 3 个脂质先验（尾链 LogP/pKa/可电离N）
#   * MTR 头：2 层 MLP + ReLU + Dropout 0.1 + mean pooling
#   * 训练：MTR-ONLY（masked token/coord/dist = 0），stock 起点，30k 步
#   * batch 16 x update_freq 2（有效 32，与 v5 一致）
# 参考论文："Transformers for Molecular Property Prediction: Domain Adaptation
# Efficiently Improves Performance."
#
# 前置：v6 LMDB 已构建（4_scripts/build_v6_descriptors_lmdb.py）
#
# 用法：
#   bash run_v6_da.sh            # 30k 步
set -euo pipefail

CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
STEPS="${1:-30000}"
CONFIG="$CT/3_configs/pipeline_v6.yaml"
OUT_DIR="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6"
LOG_DIR="$CT/logs"
LOG="$LOG_DIR/pretrain_v6_da_${STEPS}.log"
LMDB="$CT/1_data/processed/ionizable_lipids/unimolv2_3d_v6/lmdb_pretrain/lmdb_manifest.json"

cd "$CT"
mkdir -p "$LOG_DIR"

if [[ ! -f "$LMDB" ]]; then
  echo "缺少 v6 LMDB manifest：$LMDB"
  echo "请先跑：conda run -n unimol2 python 4_scripts/build_v6_descriptors_lmdb.py"
  exit 1
fi

CKPT="$OUT_DIR/checkpoints/checkpoint_last.pt"
if [[ -f "$CKPT" ]] && grep -q "num_updates: $STEPS >= max_update: $STEPS" "$LOG" 2>/dev/null; then
  echo "已完成 $STEPS 步，跳过。checkpoint：$CKPT"
  exit 0
fi

echo "=== LipidFM v6 MTR-only DA：目标 $STEPS 步 ==="
echo "LMDB：$LMDB"
echo "配置：$CONFIG"
echo "输出：$OUT_DIR"
echo "日志：$LOG"

conda run -n unimol2 --no-capture-output python main.py --step pretrain \
  --config "$CONFIG" \
  --set pretrain.total_steps="$STEPS" \
  2>&1 | tee "$LOG"

echo "=== 完成。下一步：下游评估（frozen 快筛 + 微调确认）==="
echo "  LIPIDFM_CHECKPOINT=\$PWD/$OUT_DIR/checkpoints/checkpoint_1_${STEPS}.pt \\"
echo "    conda run -n unimol2 python 4_scripts/repr_benchmark.py --dataset all --seed 42"
