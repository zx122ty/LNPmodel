#!/usr/bin/env bash
# run_v5_60k_eval_queue.sh — serial downstream evaluation for v5_60k vs v5(30k).
#
# v5_60k = 干净 v5 语料，stock → 60k 步。与 v5(30k) 唯一变量 = 预训练步数。
# 核心消融：v5_60k(60k) vs v5(30k) —— 同一套 fold cache，验证"步数 2x 是否提升下游"。
# 历史 v5(30k)/stock/v4/v5e 结果已在 moltrain_results.json，此处只跑 v5_60k 新结果。
#
# 串行原因：MolTrain 每 run 持 ~50-60GB GPU，两个并发会 OOM。
#
# 用法（在 v5_60k 训练完成后）：
#   bash run_v5_60k_eval_queue.sh
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"
mkdir -p logs

V560K_CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5_60k/checkpoints/checkpoint_2_60000.pt"
# NOTE: v5_60k reaches 60k updates during epoch 002, so the final-step checkpoint
# is checkpoint_2_60000.pt (not checkpoint_1_60000.pt). Align with v5(30k) which
# used checkpoint_1_30000.pt (final step of its run).
if [[ ! -f "$V560K_CKPT" ]]; then
  echo "!!! checkpoint_2_60000.pt not found — fallback to checkpoint_last.pt"
  V560K_CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5_60k/checkpoints/checkpoint_last.pt"
fi
echo "v5_60k checkpoint: $V560K_CKPT"

# Wait until the GPU is free (v5_60k pretraining may still be running).
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

# ---- scaffold: v5_60k (multi-seed, identical folds as v5/stock/v4) ----
run finetune_v5_60k_scaffold_s42 \
  env LIPIDFM_CHECKPOINT="$V560K_CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 42 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

run finetune_v5_60k_scaffold_s1337 \
  env LIPIDFM_CHECKPOINT="$V560K_CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 1337 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

# ---- LUMI sequential protocol (combinatorial interpolation) ----
run finetune_v5_60k_seq \
  env LIPIDFM_CHECKPOINT="$V560K_CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split sequential --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

echo "ALL DONE $(date '+%F %T')"
