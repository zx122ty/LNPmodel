#!/usr/bin/env bash
# run_v6b_eval_queue.sh — serial downstream evaluation for v6b.
#
# v6b = v5 严格对照，仅描述符 32→215。核心消融：v6b vs v5（同 fold）。
# 双轨：frozen-repr 快筛（快）+ finetune 端到端确认（最终裁判）。
#
# Usage:
#   bash run_v6b_eval_queue.sh frozen      # frozen-repr 快筛（v6b + 对照）
#   bash run_v6b_eval_queue.sh finetune    # 端到端微调确认
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"
mkdir -p logs

MODE="${1:-frozen}"

V6BCKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6b/checkpoints/checkpoint_2_30000.pt"
# NOTE: v6b's 30k steps complete at the start of epoch 2 -> final step-checkpoint
# is checkpoint_2_30000.pt (NOT checkpoint1.pt, which is epoch-1 end = 30,021).
# Align with v5 (checkpoint_1_30000.pt) final-step rule.
if [[ ! -f "$V6BCKPT" ]]; then
  echo "!!! checkpoint_2_30000.pt not found — fallback to checkpoint_last.pt"
  V6BCKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6b/checkpoints/checkpoint_last.pt"
fi
echo "v6b checkpoint: $V6BCKPT"

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
  wait_gpu_free
  run repr_v6b_all \
    conda run -n unimol2 --no-capture-output python \
      4_scripts/repr_benchmark.py --dataset all --split scaffold --seed 42 \
      --bases v6b
  run repr_v6b_seq \
    conda run -n unimol2 --no-capture-output python \
      4_scripts/repr_benchmark.py --dataset all --split sequential --seed 42 \
      --bases v6b
  echo "ALL DONE $(date '+%F %T')"
elif [[ "$MODE" == "finetune" ]]; then
  # 节约时间：端到端微调只测 Sheet2 scaffold s1337（v6/v5 配对最有区分度的
  # 协议，且 v5 历史有 s1337 结果 0.673 可直接对比）
  wait_gpu_free
  run finetune_v6b_Sheet2_scaf_s1337 \
    env LIPIDFM_CHECKPOINT="$V6BCKPT" conda run -n unimol2 --no-capture-output python \
      4_scripts/finetune_moltrain.py --dataset Sheet2 --split scaffold --seed 1337 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5
  echo "ALL DONE $(date '+%F %T')"
else
  echo "unknown mode: $MODE (use 'frozen' or 'finetune')"
  exit 1
fi
