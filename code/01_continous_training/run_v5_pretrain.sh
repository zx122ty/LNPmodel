#!/usr/bin/env bash
# run_v5_pretrain.sh —— LipidFM v5 继续预训练（MTR: masked-token/coord/dist + descriptor regression）
#
# v5 语料 = 300,000 可电离脂质（generated_virtual_library_balanced.csv）
#         + 700,000 ChEMBL 通用分子（chembl_37_chemreps.txt.gz, 确定性 reservoir 采样）
# 混合比 7(general):3(ionizable lipid) 由 LMDB 内数量比例实现：每个 epoch 端到端洗漱
# 每个 batch 平均 ~70% 通用分子 —— “数据重放”/anti-forgetting 机制。
#
# 与 v4 的全部差异：
#   1. 语料 253,756 → ~1,000,000（10x）且 70% 通用（17% → 70%）
#   2. 新增 MTR 头：unimol2_mtr task/loss/model，[CLS] 回归 34 个 z-score RDKit 描述符
#   3. 构象预算：IL 24->20（领域分子密集覆盖）、ChEMBL 14->10（通用分子低成本）
#
# 用法：
#   bash run_v5_pretrain.sh 30000            # 默认段 1（30k 步）
#   bash run_v5_pretrain.sh --steps 56600    # 等 epoch 续训（与 v2/v4 对齐）
set -euo pipefail

CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
STEPS="${1:-30000}"
OUT_DIR="2_models/continued_pretrain/unimol2_lipids_84m_v5"
SDF="1_data/processed/ionizable_lipids/unimolv2_3d_v5/lipids_unimolv2_3d.sdf"
LOG_DIR="$CT/logs"
LOG="$LOG_DIR/pretrain_v5_${STEPS}.log"

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

echo "=== LipidFM v5 继续预训练（MTR）：目标 $STEPS 步 ==="
echo "语料：$SDF"
echo "输出：$OUT_DIR"
echo "日志：$LOG"

LMDB_DIR="1_data/processed/ionizable_lipids/unimolv2_3d_v5/lmdb_pretrain"
BUILD_LMDB=""
if [[ -f "$LMDB_DIR/lmdb_manifest.json" ]]; then
  echo "→ 复用已有 LMDB：$LMDB_DIR（不重建）"
else
  echo "→ 首次运行，将构建 LMDB：$LMDB_DIR"
  BUILD_LMDB="pretrain.prepare_valid_split=true"
fi
[[ -f "$CKPT" ]] && echo "→ 检测到已有 checkpoint，将自动从 checkpoint_last.pt 续训"

conda run -n unimol2 --no-capture-output python main.py --step pretrain --set \
  pretrain.train_sdf="$SDF" \
  pretrain.output_dir="$OUT_DIR" \
  pretrain.total_steps="$STEPS" \
  $BUILD_LMDB \
  2>&1 | tee "$LOG"

echo "=== 完成。下一步 ==="
echo "  验证表征（冻结 [CLS] + MLP 头）：
    LIPIDFM_CHECKPOINT=\$PWD/$OUT_DIR/checkpoints/checkpoint_best.pt \\
      conda run -n unimol2 python 4_scripts/repr_benchmark.py --dataset all --seed 42"