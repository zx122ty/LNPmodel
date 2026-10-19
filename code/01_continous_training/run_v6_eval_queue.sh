#!/usr/bin/env bash
# run_v6_eval_queue.sh — serial downstream evaluation for v6 (MTR-only DA).
#
# Phase 2 double-track evaluation:
#   1. frozen-repr 快筛: repr_benchmark (v6 vs v5 vs v5e vs v5_60k vs stock),
#      fast (minutes/model), pure representation quality — run FIRST.
#   2. full fine-tune 确认: finetune_moltrain on the most promising candidates
#      (same fold cache), only after frozen-repr looks good.
#
# Serial: MolTrain holds ~50-60 GB GPU, two concurrent runs OOM.
#
# Usage (after v6 pretraining completes):
#   bash run_v6_eval_queue.sh          # frozen-repr first
#   bash run_v6_eval_queue.sh --finetune   # full fine-tune confirmation
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"
mkdir -p logs

MODE="${1:-frozen}"

V6CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6/checkpoints/checkpoint1.pt"
# NOTE: v6's 30k steps complete inside epoch 1 (30,021 updates), so the final
# step-checkpoint naming differs: checkpoint1.pt (epoch-1 end) is the final
# model, NOT checkpoint_1_30000.pt (which does not exist for v6).
if [[ ! -f "$V6CKPT" ]]; then
  echo "!!! checkpoint1.pt not found — fallback to checkpoint_last.pt"
  V6CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6/checkpoints/checkpoint_last.pt"
fi
echo "v6 checkpoint: $V6CKPT"

wait_gpu_free() {
  echo "===== waiting for GPU to be free $(date '+%T')"
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -lt 1000 ]; then echo "GPU free ($used MiB)"; return; fi
    sleep 120
  done
}

run() {
  local name="$1"; shift
  echo "===== [$name] $(date '+%F %T') -> $*"
  "$@" > "logs/$name.log" 2>&1 || { echo "!!!! [$name] FAILED $(date '+%T')"; }
  echo "===== [$name] done $(date '+%T')"
}

if [[ "$MODE" == "frozen" ]]; then
  # ---- frozen-repr 快筛: all bases on both sheets, scaffold + sequential ----
  wait_gpu_free
  run repr_v6_all \
    conda run -n unimol2 --no-capture-output python \
      4_scripts/repr_benchmark.py --dataset all --split scaffold --seed 42 \
      --bases stock-84M v4 v5 v5e v5_60k v6
  run repr_v6_seq \
    conda run -n unimol2 --no-capture-output python \
      4_scripts/repr_benchmark.py --dataset all --split sequential --seed 42 \
      --bases stock-84M v4 v5 v5e v5_60k v6
  echo "ALL DONE $(date '+%F %T')"
elif [[ "$MODE" == "finetune" ]]; then
  # ---- full fine-tune 确认 (scaffold + sequential, 2 seeds) ----
  wait_gpu_free
  run finetune_v6_scaffold_s42 \
    env LIPIDFM_CHECKPOINT="$V6CKPT" conda run -n unimol2 --no-capture-output python \
      4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 42 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5
  run finetune_v6_scaffold_s1337 \
    env LIPIDFM_CHECKPOINT="$V6CKPT" conda run -n unimol2 --no-capture-output python \
      4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 1337 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5
  run finetune_v6_seq \
    env LIPIDFM_CHECKPOINT="$V6CKPT" conda run -n unimol2 --no-capture-output python \
      4_scripts/finetune_moltrain.py --dataset all --split sequential --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5
  echo "ALL DONE $(date '+%F %T')"
else
  echo "unknown mode: $MODE (use 'frozen' or 'finetune')"
  exit 1
fi
