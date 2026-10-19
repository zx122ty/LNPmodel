#!/usr/bin/env bash
# run_v6d_eval_queue.sh — serial downstream evaluation for v6d (MTR head 论文式).
#
# v6d = v5 严格对照，仅 MTR head（2层MLP+ReLU+Dropout0.1+mean pooling）。
# 核心消融：v6d vs v5（同 fold，Sheet2 scaffold s1337 优先）。
# 节约模式：frozen 快筛全跑（分钟级）+ finetune 只测 Sheet2 scaffold s1337。
#
# Usage:
#   bash run_v6d_eval_queue.sh frozen      # frozen-repr 快筛
#   bash run_v6d_eval_queue.sh finetune    # 端到端微调确认（Sheet2 scaf s1337）
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"
mkdir -p logs

MODE="${1:-frozen}"

V6CCKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6d/checkpoints/checkpoint_last.pt"
# 训练完成后查日志确认 final checkpoint 命名（30k 落在 epoch 1 → checkpoint1.pt，
# 落在 epoch 2 → checkpoint_2_30000.pt；用 checkpoint_last.pt 兜底）
for cand in checkpoint_1_30000.pt checkpoint_2_30000.pt checkpoint1.pt; do
  if [[ -f "$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6d/checkpoints/$cand" ]]; then
    V6CCKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v6d/checkpoints/$cand"
    break
  fi
done
echo "v6d checkpoint: $V6CCKPT"

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
  run repr_v6d_all \
    conda run -n unimol2 --no-capture-output python \
      4_scripts/repr_benchmark.py --dataset all --split scaffold --seed 42 \
      --bases v6d
  run repr_v6d_seq \
    conda run -n unimol2 --no-capture-output python \
      4_scripts/repr_benchmark.py --dataset all --split sequential --seed 42 \
      --bases v6d
  echo "ALL DONE $(date '+%F %T')"
elif [[ "$MODE" == "finetune" ]]; then
  wait_gpu_free
  run finetune_v6d_Sheet2_scaf_s1337 \
    env LIPIDFM_CHECKPOINT="$V6CCKPT" conda run -n unimol2 --no-capture-output python \
      4_scripts/finetune_moltrain.py --dataset Sheet2 --split scaffold --seed 1337 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5
  echo "ALL DONE $(date '+%F %T')"
else
  echo "unknown mode: $MODE (use 'frozen' or 'finetune')"
  exit 1
fi
