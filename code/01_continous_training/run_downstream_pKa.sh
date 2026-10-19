#!/usr/bin/env bash
# run_downstream_pKa.sh — evaluation 2 (apparent pKa) fine-tuning queue.
#
# Protocol (user decision #2, plan 26-09-16 §1.3): 5 partition seeds x 5 folds
# = 25 folds, identical folds for every model so the comparison can be paired.
#
# Splits (user decision 2026-09-18): butina is the MAIN protocol, scaffold is the
# appendix. Both are valid for pKa -- scaffold's family-concentration ratio is
# 0.50 (largest scaffold 57 vs mean fold 115), well under the 1.0 failure
# threshold, and its measured R2 was +0.45. See prompt/26-09-16/04-*.
#
# Folds come from 5_outputs/downstream_splits/ (frozen, hashed, published) rather
# than 5_outputs/split_cache/. `--butina-cutoff 0.8` is pKa's own value: 0.7 is
# Sheet2's and on this library produces a fold holding 54.7% of the data.
#
# One run per GPU: each MolTrain process holds tens of GB, so two on one card OOM
# (measured 2026-08-29). Three RTX 6000D are available; jobs are dealt round-robin.
#
# Usage:
#   ./run_downstream_pKa.sh                 # all 10 runs (LipidFM v5)
#   MODELS=stock ./run_downstream_pKa.sh    # stock-84M ablation instead
#   SPLITS=butina ./run_downstream_pKa.sh   # main protocol only
#   FORCE=1 SPLITS=butina ./run_downstream_pKa.sh   # redo even if the cell exists
#
# 2026-09-18 note: the butina splits were rebuilt because chem_splits.butina_clusters
# had been supplying the distance vector in the wrong triangle (see
# prompt/26-09-18/06-发现-Butina距离向量顺序错误.md). The old folds, run dirs,
# result cells and bundle are quarantined under
# failed/butina_distance_order_bug_20260918/. Every pKa butina number must come from
# the corrected folds in 5_outputs/downstream_splits/. scaffold was never affected.
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"
mkdir -p logs/downstream_pKa

V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
MODELS="${MODELS:-v5}"
SPLITS="${SPLITS:-butina scaffold}"
SEEDS="${SEEDS:-42 123 456 789 1011}"
GPUS="${GPUS:-0 1 2}"

# Same hyperparameters as the historical Sheet2 runs, so the pKa number is
# comparable to them: lr 5e-5, 40 epochs, early-stopping 8, batch 16 (default).
EPOCHS=40; LR=5e-5; ES=8

case "$MODELS" in
  v5)    export LIPIDFM_CHECKPOINT="$V5CKPT"; TAG=v5 ;;
  stock) export USE_STOCK_84M=1;              TAG=stock ;;
  *) echo "MODELS must be v5 or stock (got '$MODELS')" >&2; exit 2 ;;
esac

# --- idempotence guard (added 2026-09-18, closes gap #1 of 05-pKa评估结果.md §9.1) ---
# MolTrain's save_path has NO timestamp, so re-running a job overwrites cv.data in
# place while the result JSON keeps the first run's numbers -- two artifacts that are
# each internally consistent and whose cross-check passes. That happened once (report
# §8) and was survived only by luck. So: skip a job whose collected result cell
# already exists, unless FORCE=1.
#
# The guard is deliberately on the COLLECTED cell (downstream_results/), not on the
# run directory: the run directory is written by MolTrain itself and is the thing that
# gets clobbered, so it cannot serve as evidence that a job finished.
#
# The cell's model label comes from `collect_neural.py --model`, NOT from $TAG
# ($TAG is "v5"/"stock"; the cells say "lipidfm_v5"/"stock_84m"). Mapping them here
# rather than reusing $TAG, because a guard that looks for the wrong filename silently
# never fires -- which is the same class of failure it exists to prevent.
RESULTS_DIR="5_outputs/downstream_results"
case "$MODELS" in
  v5)    ARM_LABEL=lipidfm_v5 ;;
  stock) ARM_LABEL=stock_84m ;;
  *)     echo "MODELS must be v5 or stock (got '$MODELS')" >&2; exit 2 ;;
esac
is_done() {  # $1=split $2=seed
  [ -s "$RESULTS_DIR/pKa|${ARM_LABEL}|$1|seed$2|k5.json" ]
}

# Build the job list: one entry per (split, seed).
JOBS=()
for S in $SPLITS; do
  for SD in $SEEDS; do
    if [ "${FORCE:-0}" != "1" ] && is_done "$S" "$SD"; then
      echo "=== SKIP pKa/$S/seed$SD -- result cell already exists (FORCE=1 to redo) ==="
      continue
    fi
    JOBS+=("$S:$SD")
  done
done
echo "=== ${#JOBS[@]} jobs, model=$TAG, gpus=[$GPUS], $(date '+%F %T') ==="
if [ ${#JOBS[@]} -eq 0 ]; then echo "nothing to do (all result cells present)"; exit 0; fi
printf '    %s\n' "${JOBS[@]}"

# Deal jobs to GPUs round-robin, one sequential worker per GPU.
i=0
for G in $GPUS; do
  (
    for j in "${!JOBS[@]}"; do
      [ $(( j % $(echo $GPUS | wc -w) )) -ne $i ] && continue
      IFS=: read -r SPLIT SEED <<< "${JOBS[$j]}"
      NAME="finetune_${TAG}_pKa_${SPLIT}_s${SEED}"
      echo "===== [$NAME] gpu$G $(date '+%F %T')"
      env CUDA_VISIBLE_DEVICES="$G" conda run -n unimol2 --no-capture-output python \
        4_scripts/finetune_moltrain.py \
        --dataset pKa --split "$SPLIT" --butina-cutoff 0.8 \
        --seed "$SEED" --lr "$LR" --epochs "$EPOCHS" --early-stopping "$ES" --kfold 5 \
        > "logs/downstream_pKa/${NAME}.log" 2>&1 \
        || echo "!!!! [$NAME] FAILED $(date '+%T') (see logs/downstream_pKa/${NAME}.log)"
      echo "===== [$NAME] done $(date '+%F %T')"
    done
  ) &
  i=$(( i + 1 ))
done
wait
echo "ALL DONE $(date '+%F %T')"
