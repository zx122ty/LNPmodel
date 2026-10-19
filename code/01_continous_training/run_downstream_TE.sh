#!/usr/bin/env bash
# run_downstream_TE.sh — evaluation 1 (TE, transfection efficiency) fine-tuning queue.
#
# Protocol (user decision #2, plan 26-09-16 §1.3): the plan grid
# {42, 123, 456, 789, 1011} x 5 folds = 25 folds per (dataset, protocol), identical
# folds for every model so the comparison can be paired.
#
# Splits: BOTH protocols for BOTH datasets.
#   * Sheet2 — scaffold is the historical main protocol ("照旧 scaffold"), and
#     butina is re-run here because the published butina column (0.7649 / 0.7887 /
#     0.8121, CLAUDE.md front page) was computed on folds built by the broken
#     Butina splitter (坑 #14) and has to be replaced.
#   * Sheet3 — butina is the main protocol, scaffold the appendix (user decision
#     2026-09-18): Sheet3's scaffold family-concentration ratio is 1.14, over the
#     1.0 failure threshold, and its measured R2 was -0.83. See 26-09-16/04-*.
#
# Not run here: `sequential` (historical protocol, not part of the three-task plan)
# and `random` (evaluation 3's main protocol, not evaluation 1's).
#
# Folds come from 5_outputs/downstream_splits/ (frozen, hashed, published). Sheet2
# and Sheet3 are `frozen_split` datasets, so finetune_moltrain reads the frozen file
# and does NOT rebuild from split_cache. The frozen store carries 7 seeds for
# scaffold/butina -- the plan grid plus the historical {1337, 2024} -- so that every
# published queue script can still be re-run verbatim. Only the plan grid is run by
# default: 7 seeds would be 35 folds, and the headline is specified as 25 so that
# evaluation 1 is paired fold-for-fold with pKa and MTR. Pass SEEDS= to add them.
#
# One run per GPU: each MolTrain process holds tens of GB, so two on one card OOM
# (measured 2026-08-29). Three RTX 6000D are available; jobs are dealt round-robin.
#
# Measured per-fold wall time (from model_*.pth mtimes, 3-GPU-contended):
#   Sheet2 26.4 min/fold, Sheet3 10.6 min/fold  (molecules 1920 vs 768)
#   => 2 protocols x 5 seeds x 5 folds x {26.4, 10.6} = 22.0 + 8.8 = ~30.8 GPU·h
#   => ~10 h wall on 3 GPUs. CPU budget for the baseline arms is 25.9 cpu·h
#      (measured, cost_probe.py; see 26-09-18/05-* §9.4).
#
# Usage:
#   DRY_RUN=1 ./run_downstream_TE.sh                 # print the job list, launch nothing
#   ./run_downstream_TE.sh                           # both datasets, LipidFM v5
#   DATASET=Sheet3 ./run_downstream_TE.sh            # one dataset
#   MODELS=stock ./run_downstream_TE.sh              # stock-84M ablation instead
#   SPLITS=butina ./run_downstream_TE.sh             # main protocol only
#   FORCE=1 SPLITS=butina ./run_downstream_TE.sh     # redo even if the cell exists
#
# 2026-09-18 note: the Sheet2/Sheet3 butina folds were rebuilt because
# chem_splits.butina_clusters had been supplying the distance vector in the wrong
# triangle (see prompt/26-09-18/06-发现-Butina距离向量顺序错误.md). The old folds,
# run dirs, result cells and the aggregate entries computed on them are quarantined
# or marked void under failed/butina_distance_order_bug_20260918/. Every butina
# number from here on comes from the corrected frozen folds. scaffold was never
# affected.
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"

V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
MODELS="${MODELS:-v5}"
# DATASET (singular) is accepted as an alias because that is the name every other
# script in this project uses (`--dataset`). A queue script whose variable silently
# does nothing is worse than one that errors.
DATASETS="${DATASETS:-${DATASET:-Sheet2 Sheet3}}"
SPLITS="${SPLITS:-scaffold butina}"
SEEDS="${SEEDS:-42 123 456 789 1011}"
GPUS="${GPUS:-0 1 2}"
LOGDIR="${LOGDIR:-logs/downstream_TE}"
mkdir -p "$LOGDIR"

# Same hyperparameters as the historical Sheet2 runs, so the new numbers are
# comparable to the published ones: lr 5e-5, 40 epochs, early-stopping 8, batch 16.
EPOCHS=40; LR=5e-5; ES=8

case "$MODELS" in
  v5)    export LIPIDFM_CHECKPOINT="$V5CKPT"; TAG=v5 ;;
  stock) export USE_STOCK_84M=1;              TAG=stock ;;
  *) echo "MODELS must be v5 or stock (got '$MODELS')" >&2; exit 2 ;;
esac

# --- idempotence guard (same pattern as run_downstream_pKa.sh) -------------------
# MolTrain's save_path has NO timestamp, so re-running a job overwrites cv.data in
# place while the result JSON keeps the first run's numbers -- two artifacts that are
# each internally consistent and whose cross-check passes. So: skip a job whose
# collected result cell already exists, unless FORCE=1.
#
# The guard is on the COLLECTED cell (downstream_results/), not on the run directory:
# the run directory is written by MolTrain itself and is the thing that gets
# clobbered, so it cannot serve as evidence that a job finished. That is also why
# this script collects its own runs (see below) instead of leaving collection as a
# separate manual step -- a guard whose signal is produced by a different script than
# its consumer can silently never fire.
#
# The cell's model label comes from `collect_neural.py --model`, NOT from $TAG
# ($TAG is "v5"/"stock"; the cells say "lipidfm_v5"/"stock_84m"). Mapping them here
# rather than reusing $TAG, because a guard that looks for the wrong filename
# silently never fires -- which is the same class of failure it exists to prevent.
case "$MODELS" in
  v5)    ARM_LABEL=lipidfm_v5 ;;
  stock) ARM_LABEL=stock_84m ;;
  *)     echo "MODELS must be v5 or stock (got '$MODELS')" >&2; exit 2 ;;
esac
RESULTS_DIR="5_outputs/downstream_results"
is_done() {  # $1=dataset $2=split $3=seed
  [ -s "$RESULTS_DIR/$1|${ARM_LABEL}|$2|seed$3|k5.json" ]
}

# Build the job list: one entry per (dataset, split, seed).
JOBS=()
for DS in $DATASETS; do
  for S in $SPLITS; do
    for SD in $SEEDS; do
      if [ "${FORCE:-0}" != "1" ] && is_done "$DS" "$S" "$SD"; then
        echo "=== SKIP $DS/$S/seed$SD -- result cell already exists (FORCE=1 to redo) ==="
        continue
      fi
      JOBS+=("$DS:$S:$SD")
    done
  done
done
NJOBS=${#JOBS[@]}
echo "=== $NJOBS jobs, model=$TAG, datasets=[$DATASETS], splits=[$SPLITS], seeds=[$SEEDS] ==="
echo "=== gpus=[$GPUS], $(date '+%F %T') ==="
if [ "$NJOBS" -eq 0 ]; then echo "nothing to do (all result cells present)"; exit 0; fi
printf '    %s\n' "${JOBS[@]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=== DRY_RUN=1 -- nothing launched. Run without DRY_RUN to execute. ==="
  exit 0
fi

# Deal jobs to GPUs round-robin, one sequential worker per GPU.
NGPU=$(echo $GPUS | wc -w)
i=0
for G in $GPUS; do
  (
    for j in "${!JOBS[@]}"; do
      [ $(( j % NGPU )) -ne $i ] && continue
      IFS=: read -r DS SPLIT SEED <<< "${JOBS[$j]}"
      NAME="finetune_${TAG}_${DS}_${SPLIT}_s${SEED}"
      echo "===== [$NAME] gpu$G $(date '+%F %T')"
      if env CUDA_VISIBLE_DEVICES="$G" conda run -n unimol2 --no-capture-output python \
          4_scripts/finetune_moltrain.py \
          --dataset "$DS" --split "$SPLIT" --butina-cutoff 0.7 \
          --seed "$SEED" --lr "$LR" --epochs "$EPOCHS" --early-stopping "$ES" --kfold 5 \
          > "$LOGDIR/${NAME}.log" 2>&1 ; then
        # Collect immediately, while this run is unambiguously the newest matching
        # result file. `--pick-newest` is only sound here for exactly that reason:
        # on Sheet2/scaffold the folds never changed, so a dozen historical runs
        # legitimately share the protocol key and the finder cannot tell them apart
        # by identity alone. It prints every candidate it passed over.
        #
        # Without this step the guard above would never see a cell and every
        # re-launch would re-train everything from scratch.
        if conda run -n unimol2 --no-capture-output python \
            4_scripts/downstream/collect_neural.py \
            --dataset "$DS" --split "$SPLIT" --seed "$SEED" --model "$ARM_LABEL" \
            --lr "$LR" --epochs "$EPOCHS" --pick-newest \
            >> "$LOGDIR/${NAME}.log" 2>&1 ; then
          echo "===== [$NAME] done $(date '+%F %T')"
        else
          echo "!!!! [$NAME] TRAINED BUT COLLECT FAILED $(date '+%T') (see $LOGDIR/${NAME}.log)"
        fi
      else
        echo "!!!! [$NAME] FAILED $(date '+%T') (see $LOGDIR/${NAME}.log)"
      fi
    done
  ) &
  i=$(( i + 1 ))
done
wait
echo "ALL DONE $(date '+%F %T')"
echo "Next: paired table per (dataset, protocol), then the 3 classical baselines on the"
echo "same frozen folds --"
echo "  python 4_scripts/downstream/report_table.py --dataset Sheet2 --split scaffold \\"
echo "    --models lipidfm_v5 ecfp4_rf ecfp4_xgb mlp_morgan"
echo "  python 4_scripts/downstream/run_downstream_baselines.py \\"
echo "    --dataset Sheet2 --split scaffold --models ecfp4_rf ecfp4_xgb mlp_morgan"
