#!/usr/bin/env bash
# run_v4_pretrain.sh —— LipidFM v4 继续预训练（分段：先 30k 验证，再续到 56.6k）
#
# 为什么分两段
# ============
# v4 语料 = v2 base + 119,230 个「骨架分层 + 真实画像重加权」采样的新虚拟脂质，
# 规模与 v3 严格相同（231,611 分子），唯一变量是**多样性**：
#
#   指标                     真实脂质   v3新增   v4新增
#   ECFP4 内部多样性          0.727     0.515    0.696
#   → Sheet2 下游 NN 中位     0.623     0.521    0.672
#   唯一 Murcko 骨架          294       12       1060
#   n_rings 中位              0         1        1
#
# 段 1（30,000 步）：与 v3 **同步数、同语料量**，是最干净的对照——
#     v3 vs v4 的差异只能来自语料多样性。跑完先用 repr_benchmark.py 分钟级验证。
# 段 2（56,600 步）：与 v2 **等 epoch**（7.21），补完 scaling 曲线的公平比较。
#     v2 = 30,000×32/133,181 = 7.21 epoch；v4 = 56,600×32/251,218 = 7.21 epoch。
#     （v3 只有 3.82 epoch —— 这是此前 v2/v3 比较里一个被忽略的混淆变量。）
#
# ⚠️ 分段的已知代价（判读段 2 时务必知情）
# ------------------------------------------------
# lr-scheduler 是 polynomial_decay，终点绑定 total_steps：
#   段 1 设 total=30000 → 跑到 30k 时 lr 已衰减到 0；
#   段 2 设 total=56600 从 30k 续训 → lr 跳回 2.39e-5 再线性降到 0。
# 因此段 2 的 lr 轨迹 ≠ 一次性跑 56,600 步（后者在 30k 处 lr=2.39e-5 且从未归零），
# 相当于中途做了一次 warm restart。
#
#   → 段 1 的对照**不受影响**：它与 v3 完全同口径（同 total=30000、同样跑到 lr=0），
#     v3 vs v4 的比较是干净的。
#   → 段 2 与 v2 比较时，需注明「等 epoch 但经过一次 warm restart」。若要求 lr 轨迹
#     也严格可比，应从 stock 底座一次性跑满 56,600 步（多花 38h，不复用段 1）。
#
# 除 total_steps 外，所有超参与 v2/v3 逐项一致（lr 5e-5、batch 8×4、mask 0.15、
# loss 1/1/1、warmup 1000、seed 20260805），保证唯一变量是语料。
#
# 用法：
#   bash run_v4_pretrain.sh 30000     # 段 1
#   bash run_v4_pretrain.sh 56600     # 段 2（自动从 checkpoint_last.pt 续训）
set -euo pipefail

STEPS="${1:-30000}"
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
OUT_DIR="2_models/continued_pretrain/unimol2_lipids_84m_v4"
SDF="1_data/processed/ionizable_lipids/unimolv2_3d_v4/lipids_unimolv2_3d.sdf"
LOG_DIR="$CT/logs"
LOG="$LOG_DIR/pretrain_v4_${STEPS}.log"

cd "$CT"
mkdir -p "$LOG_DIR"

if [[ ! -f "$SDF" ]]; then
  echo "缺少 v4 SDF：$SDF"
  echo "请先跑：python 4_scripts/generate_v3_newonly_conformers.py --version v4"
  echo "    再跑：python 4_scripts/merge_v2_plus_newonly_sdf.py --version v4"
  exit 1
fi

# 完成检测：已达目标步数就不重复跑（可安全重入）
CKPT="$OUT_DIR/checkpoints/checkpoint_last.pt"
if [[ -f "$CKPT" ]] && grep -q "num_updates: $STEPS >= max_update: $STEPS" "$LOG" 2>/dev/null; then
  echo "已完成 $STEPS 步，跳过。checkpoint：$CKPT"
  exit 0
fi

echo "=== LipidFM v4 继续预训练：目标 $STEPS 步 ==="
echo "语料：$SDF"
echo "输出：$OUT_DIR"
echo "日志：$LOG"

# LMDB 只在首次构建。段 2 续训必须复用段 1 的 LMDB，
# 否则 valid 集会被重新随机划分，两段的 valid loss 不可比。
LMDB_DIR="1_data/processed/ionizable_lipids/unimolv2_3d_v4/lmdb_pretrain"
BUILD_LMDB=""
if [[ -f "$LMDB_DIR/lmdb_manifest.json" ]]; then
  echo "→ 复用已有 LMDB：$LMDB_DIR（不重建，保证 valid 集跨段一致）"
else
  echo "→ 首次运行，将构建 LMDB：$LMDB_DIR"
  BUILD_LMDB="pretrain.prepare_valid_split=true"
fi
[[ -f "$CKPT" ]] && echo "→ 检测到已有 checkpoint，unicore 将自动从 checkpoint_last.pt 续训"

conda run -n unimol2 --no-capture-output python main.py --step pretrain --set \
  pretrain.train_sdf="$SDF" \
  pretrain.output_dir="$OUT_DIR" \
  pretrain.total_steps="$STEPS" \
  $BUILD_LMDB \
  2>&1 | tee "$LOG"

echo "=== 完成。下一步 ==="
echo "  段 1 跑完 → 分钟级验证表征："
echo "    LIPIDFM_CHECKPOINT=\$PWD/$OUT_DIR/checkpoints/checkpoint_best.pt \\"
echo "      conda run -n unimol2 python 4_scripts/repr_benchmark.py --dataset all --seed 42"
echo "  表征胜出 → 续到等 epoch：bash run_v4_pretrain.sh 56600"
