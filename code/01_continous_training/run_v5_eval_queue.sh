#!/usr/bin/env bash
# run_v5_eval_queue.sh — serial downstream evaluation for v5 vs stock vs v4.
#
# Why serial: each MolTrain run holds ~50-60 GB GPU; two at once OOMs (2026-08-29).
# This script runs the queued commands one at a time, logging each, so the GPU is
# never idle and nothing needs babysitting.
#
# Queue (scaffold, multi-seed, identical folds -> paired comparison):
#   v5    seed42  Sheet2+Sheet3  (done 2026-08-29; Sheet3 s42 was OOM'd -> re-run below)
#   v5    seed42  Sheet3         (in progress now, separate log)
#   v4    seed42  all
#   v4    seed1337 all
#   stock seed42  all
#   stock seed1337 all
# Then the LUMI sequential protocol for all three bases (seed-agnostic).
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"
mkdir -p logs

V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
V4CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v4/checkpoints/checkpoint_best.pt"

# Wait until the GPU is free (a Sheet3 s42 run may still be finishing).
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

# ---- scaffold: v4 (the two v5 scaffold seeds are done/being done) ----
run finetune_v4_scaffold_s42 \
  env LIPIDFM_CHECKPOINT="$V4CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 42 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

run finetune_v4_scaffold_s1337 \
  env LIPIDFM_CHECKPOINT="$V4CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 1337 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

# ---- scaffold: stock ----
run finetune_stock_scaffold_s42 \
  env USE_STOCK_84M=1 conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 42 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

run finetune_stock_scaffold_s1337 \
  env USE_STOCK_84M=1 conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split scaffold --seed 1337 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

# ---- LUMI sequential protocol (combinatorial interpolation) for all bases ----
run finetune_v5_seq \
  env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split sequential --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

run finetune_v4_seq \
  env LIPIDFM_CHECKPOINT="$V4CKPT" conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split sequential --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

run finetune_stock_seq \
  env USE_STOCK_84M=1 conda run -n unimol2 --no-capture-output python \
    4_scripts/finetune_moltrain.py --dataset all --split sequential --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5

echo "ALL DONE $(date '+%F %T')"
