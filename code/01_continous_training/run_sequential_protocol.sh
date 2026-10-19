#!/usr/bin/env bash
# End-to-end finetuning under LUMI-lab's own protocol (--split sequential).
#
# Same hyperparameters as the scaffold runs (lr 5e-5 / 40ep / early-stop 8), so
# protocol is the ONLY variable vs the archived scaffold numbers:
#   stock 0.6624 | v2 0.6514 | v3 0.6589 | v4 0.6403
# Skips any base whose result is already aggregated (safe to re-run).
set -u
cd "$(dirname "$0")"
LOG=logs/sequential_$(date +%Y%m%d); mkdir -p "$LOG"
COMMON="--dataset Sheet2 --split sequential --seed 42 --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5"
AGG=5_outputs/finetune_moltrain/moltrain_results.json

done_already() {  # $1 = run_key substring
  [ -f "$AGG" ] && python - "$1" <<'PY' 2>/dev/null
import json,sys
d=json.load(open("5_outputs/finetune_moltrain/moltrain_results.json"))
sys.exit(0 if any(sys.argv[1] in k for k in d) else 1)
PY
}

for base in stock v2 v3 v4; do
  key="Sheet2|sequential|seed42|lr5e-05|ep40"
  case $base in
    stock) label="stock-84M";      env_pfx="USE_STOCK_84M=1" ;;
    *)     label="unimol2_lipids_84m_${base}"
           env_pfx="LIPIDFM_CHECKPOINT=$PWD/2_models/continued_pretrain/unimol2_lipids_84m_${base}/checkpoints/checkpoint_best.pt" ;;
  esac
  if done_already "${label}|${key}"; then
    echo "[skip] ${label} already has ${key}"; continue
  fi
  echo "[run ] ${label} $(date +%H:%M:%S)"
  env $env_pfx conda run -n unimol2 python 4_scripts/finetune_moltrain.py $COMMON \
      > "$LOG/${base}.log" 2>&1
  tail -6 "$LOG/${base}.log" | grep -E "fold-mean|relaxed" || echo "  (see $LOG/${base}.log)"
done
echo "ALL DONE $(date +%H:%M:%S)"
