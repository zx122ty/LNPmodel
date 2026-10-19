#!/usr/bin/env bash
# run_v5e_eval_queue.sh — serial downstream evaluation for v5e vs v5 (vs stock/v4).
#
# v5e = v5 语料 + 2679 下游评估分子（Domain-Adaptive Pre-training）。
# 核心消融：v5e（语料含 eval） vs v5（语料干净）——同一套 fold cache，唯一变量 = +2679 分子。
# v5 / stock / v4 的历史结果已存在（moltrain_results.json），此处只跑 v5e 新结果。
#
# 串行原因：MolTrain 每 run 持 ~50-60 GB GPU，两个并发会 OOM（2026-08-29）。
#
# 用法（在 v5e 训练完成后）：
#   bash run_v5e_eval_queue.sh
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"
mkdir -p logs

V5ECKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5e/checkpoints/checkpoint_1_30000.pt"
# NOTE: historical v5/stock/v4 eval used checkpoint_1_30000.pt (final step), NOT
# checkpoint_best.pt — keep v5e aligned for a fair v5e-vs-v5 comparison.
if [[ ! -f "$V5ECKPT" ]]; then
  V5ECKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5e/checkpoints/checkpoint_best.pt"
  echo "!!! checkpoint_1_30000.pt not found — falling back to checkpoint_best.pt"
fi
echo "v5e checkpoint: $V5ECKPT"

# Wait until the GPU is free (v5e pretraining may still be running).
wait_gpu_free() {
  echo "===== waiting for GPU to be free $(date '+%T')"
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -lt 1000 ]; then echo "GPU free ($used MiB)"; return; fi
    sleep 120
  done
}
wait_gpu_free

run() {
  local name="$1"; shift
  echo "===== [$name] $(date '+%F %T') -> $*"
  # shellcheck disable=SC2068
  "$@" > "logs/$name.log" 2>&1 || { echo "!!!! [$name] FAILED $(date '+%T')"; }
  echo "===== [$name] done $(date '+%T')"
}

# ---- scaffold: v5e (multi-seed, identical folds as v5/stock/v4) ----
run finetune_v5e_scaffold_s42 \
  env LIPIDFM_CHECKPOINT="$V5ECKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 42 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

run finetune_v5e_scaffold_s1337 \
  env LIPIDFM_CHECKPOINT="$V5ECKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 1337 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

# ---- LUMI sequential protocol (combinatorial interpolation) ----
run finetune_v5e_seq \
  env LIPIDFM_CHECKPOINT="$V5ECKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split sequential --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

echo "ALL DONE $(date '+%F %T')"
