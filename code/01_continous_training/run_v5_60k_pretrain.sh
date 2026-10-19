#!/usr/bin/env bash
# run_v5_60k_pretrain.sh —— LipidFM v5_60k 预训练（MTR，60k 步）
#
# v5_60k = 干净 v5 语料（300k 脂质 + 700k ChEMBL），从 stock 84M 全新训 60k 步。
# 目的：验证"预训练步数 2x（30k→60k）是否提升下游"（方向 A 消融）。
# 与 v5(30k) 唯一变量 = total_steps 60000 vs 30000；其余超参逐项一致。
# 注意：语料是干净的 v5（不含下游 eval 分子）—— v5e 那套 DA-PT 消融与此独立。
#
# 用法：
#   bash run_v5_60k_pretrain.sh            # 60k 步（stock 全重训）
set -euo pipefail

CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
STEPS="${1:-60000}"
CONFIG="$CT/3_configs/pipeline_v5_60k.yaml"
OUT_DIR="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5_60k"
LOG_DIR="$CT/logs"
LOG="$LOG_DIR/pretrain_v5_60k_${STEPS}.log"
SDF="$CT/1_data/processed/ionizable_lipids/unimolv2_3d_v5/lipids_unimolv2_3d.sdf"

cd "$CT"
mkdir -p "$LOG_DIR"

if [[ ! -f "$SDF" ]]; then
  echo "缺少 v5 SDF：$SDF"
  echo "请先跑：conda run -n unimol2 python 4_scripts/build_v5_dataset.py"
  echo "    再跑：conda run -n unimol2 python 4_scripts/build_v5_descriptors_lmdb.py"
  exit 1
fi

CKPT="$OUT_DIR/checkpoints/checkpoint_last.pt"
if [[ -f "$CKPT" ]] && grep -q "num_updates: $STEPS >= max_update: $STEPS" "$LOG" 2>/dev/null; then
  echo "已完成 $STEPS 步，跳过。checkpoint：$CKPT"
  exit 0
fi

echo "=== LipidFM v5_60k 预训练（MTR，stock 全重训 $STEPS 步）==="
echo "语料：$SDF（干净 v5，不含下游 eval）"
echo "配置：$CONFIG"
echo "输出：$OUT_DIR"
echo "日志：$LOG"

conda run -n unimol2 --no-capture-output python main.py --step pretrain \
  --config "$CONFIG" \
  --set pretrain.total_steps="$STEPS" \
  2>&1 | tee "$LOG"

echo "=== 完成。下一步：下游评估 ==="
echo "  LIPIDFM_CHECKPOINT=\$PWD/$OUT_DIR/checkpoints/checkpoint_1_${STEPS}.pt \\"
echo "    conda run -n unimol2 python 4_scripts/finetune_moltrain.py ..."
