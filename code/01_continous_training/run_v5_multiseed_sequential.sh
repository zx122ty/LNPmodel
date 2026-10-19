#!/usr/bin/env bash
# Validation gate A: does LipidFM-v4's apparent SEQ-protocol edge survive multiple seeds?
# LUMI sequential split is deterministic (seed-free); swapping --seed changes only MolTrain
# training RNG. We re-validate v4 vs stock on Sheet2+Sheet3 at seeds 1337/2026/515.
# Existing seed42 seq runs are already in moltrain_results.json as an anchor.
#
# Each run writes its own timestamped JSON + its own exp dir -> no overwrite.
set -euo pipefail

CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
V4="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v4/checkpoints/checkpoint_best.pt"
LOG_DIR="$CT/logs"
mkdir -p "$LOG_DIR"
cd "$CT"

run() {
  local base="$1" seed="$2"
  local tag="${base}_seq_s${seed}"
  local log="$LOG_DIR/finetune_${tag}.log"
  echo ""
  echo "################  $tag  :  $(date)  ################"
  if [[ "$base" == "stock" ]]; then
    USE_STOCK_84M=1 conda run -n unimol2 --no-capture-output \
      python 4_scripts/finetune_moltrain.py --dataset all --split sequential \
        --seed "$seed" --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 2>&1 | tee "$log"
  else
    LIPIDFM_CHECKPOINT="$V4" conda run -n unimol2 --no-capture-output \
      python 4_scripts/finetune_moltrain.py --dataset all --split sequential \
      --seed "$seed" --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 2>&1 | tee "$log"
  fi
  echo "### finished $base seed $seed : exit=$? ($(date))"
}

echo "=== Gate-A multiseed sequential validation started $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')"

run stock 1337
run v4    1337
run v4    515
run v4    2026
run stock 2026

echo ""
echo "=== ALL DONE $(date) ==="