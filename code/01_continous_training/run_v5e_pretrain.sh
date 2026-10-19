#!/usr/bin/env bash
# run_v5e_pretrain.sh —— LipidFM v5e 继续预训练（MTR）
#
# v5e = v5 语料 + 下游评估分子（Domain-Adaptive Pre-training）：
#   302,679 可电离脂质/评估分子（generated_virtual_library_balanced_v2.csv）
#   + 700,000 ChEMBL 通用分子
# 从 stock 84M 全重训 30k 步，超参与 v5 完全一致（唯一变量 = +2679 eval 分子）。
#
# 用法：
#   bash run_v5e_pretrain.sh            # 30k 步（stock 全重训）
set -euo pipefail

CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
STEPS="${1:-30000}"
CONFIG="$CT/3_configs/pipeline_v5e.yaml"
OUT_DIR="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5e"
LOG_DIR="$CT/logs"
LOG="$LOG_DIR/pretrain_v5e_${STEPS}.log"
SDF="$CT/1_data/processed/ionizable_lipids/unimolv2_3d_v5e/lipids_unimolv2_3d.sdf"

cd "$CT"
mkdir -p "$LOG_DIR"

if [[ ! -f "$SDF" ]]; then
  echo "缺少 v5e SDF：$SDF"
  echo "请先跑：conda run -n unimol2 python 4_scripts/build_v5_eval_incremental.py"
  exit 1
fi

CKPT="$OUT_DIR/checkpoints/checkpoint_last.pt"
if [[ -f "$CKPT" ]] && grep -q "num_updates: $STEPS >= max_update: $STEPS" "$LOG" 2>/dev/null; then
  echo "已完成 $STEPS 步，跳过。checkpoint：$CKPT"
  exit 0
fi

echo "=== LipidFM v5e 继续预训练（MTR，stock 全重训 $STEPS 步）==="
echo "语料：$SDF"
echo "配置：$CONFIG"
echo "输出：$OUT_DIR"
echo "日志：$LOG"

conda run -n unimol2 --no-capture-output python main.py --step pretrain \
  --config "$CONFIG" \
  --set pretrain.total_steps="$STEPS" \
  2>&1 | tee "$LOG"

echo "=== 完成。下一步：下游评估 ==="
echo "  LIPIDFM_CHECKPOINT=\$PWD/$OUT_DIR/checkpoints/checkpoint_best.pt \\"
echo "    conda run -n unimol2 python 4_scripts/finetune_moltrain.py ..."
