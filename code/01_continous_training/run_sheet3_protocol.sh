#!/usr/bin/env bash
# End-to-end finetuning on Sheet3 (n=768), BOTH protocols (scaffold + sequential).
#
# Model = log10(transfection_efficiency), standardized inside MolTrain.
# Same hyperparams as all prior runs (lr 5e-5 / 40ep / early-stop 8) so the only
# thing that differs from Sheet2 is the dataset.
#
# IMPORTANT — no overwrites:
#   * each run writes to its own timestamped moltrain_results_<ts>.json (default)
#   * exp_tag dir is keyed by base+ds+split+seed+lr+ep (finetune_moltrain), so
#     Sheet3 runs live under distinct directories from the Sheet2 archives.
#   * the shared aggregate moltrain_results.json uses full-dimension run keys, so
#     Sheet3 keys ("...|Sheet3|...") never clobber Sheet2 keys.
set -u
cd "$(dirname "$0")"
LOG=logs/sheet3_$(date +%Y%m%d); mkdir -p "$LOG"
AGG=5_outputs/finetune_moltrain/moltrain_results.json

done_already() {  # $1 = run_key substring
  [ -f "$AGG" ] && python - "$1" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open("5_outputs/finetune_moltrain/moltrain_results.json"))
    runs=d.get("runs", d)
except Exception:
    sys.exit(1)
sys.exit(0 if any(sys.argv[1] in k for k in runs) else 1)
PY
}

for split in scaffold sequential; do
  for base in stock v2 v3 v4; do
    key="Sheet3|${split}|seed42|lr5e-05|ep40"
    case $base in
      stock) label="stock-84M";      env_pfx="USE_STOCK_84M=1" ;;
      *)     label="unimol2_lipids_84m_${base}"
             env_pfx="LIPIDFM_CHECKPOINT=$PWD/2_models/continued_pretrain/unimol2_lipids_84m_${base}/checkpoints/checkpoint_best.pt" ;;
    esac
    if done_already "${label}|${key}"; then
      echo "[skip] ${label} ${split} already has ${key}"
      continue
    fi
    echo "[run ] ${label} split=${split} $(date +%H:%M:%S)"
    env $env_pfx conda run -n unimol2 python 4_scripts/finetune_moltrain.py \
        --dataset Sheet3 --split ${split} --seed 42 --lr 5e-5 --epochs 40 \
        --early-stopping 8 --kfold 5 \
        > "$LOG/${base}_${split}.log" 2>&1
    tail -6 "$LOG/${base}_${split}.log" | grep -E "fold-mean|relaxed" || echo "  (see $LOG/${base}_${split}.log)"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"