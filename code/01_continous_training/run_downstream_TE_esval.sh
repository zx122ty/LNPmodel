#!/usr/bin/env bash
# run_downstream_TE_esval.sh — 坑 #12 control arm (train-internal early stopping).
#
# What this is
# ------------
# MolTrain's early stopping watches the TEST fold (`nnmodel.py:188-190`:
# `X_valid, y_valid = X[te_idx], y[te_idx]`), so every published fold-mean R2 in this
# project is on a protocol that is strictly EASIER than the one the classical arms
# use (`downstream/baselines.py` carves `val_frac=0.15` out of TRAIN). 坑 #12.
#
# This queue runs the SAME model, SAME folds, SAME budget, and changes exactly one
# thing: early stopping watches a 15% train-internal holdout instead of the test fold
# (`finetune_moltrain.py --es-val internal`). Each real test fold is then re-predicted
# from `model_<k>.pth`. The result is on the SAME protocol as the classical arms, so
# it can go in the same column, and the raw gap (default minus this) is a direct
# measurement of the 坑 #12 advantage -- instead of inferring it from the mean-shift
# decomposition in `prompt/26-09-18/08-*` §3.
#
# Scope: the SAME 4 (dataset, protocol) x 5 seeds x 5 folds = 100 fold-jobs as
# run_downstream_TE.sh, so the two arms are paired fold-for-fold. Do not add a split
# here that the main arm did not run, or the pairing breaks.
#
# Cost: identical per-fold budget to the default arm, so the same ~30.8 GPU·h /
# ~10 h wall on 3 GPUs (measured 26.4 min/fold Sheet2, 10.6 min/fold Sheet3 under
# 3-GPU contention). The re-prediction adds 5 forward passes per job: seconds.
#
# Usage:
#   DRY_RUN=1 ./run_downstream_TE_esval.sh          # print the job list, launch nothing
#   ./run_downstream_TE_esval.sh                    # all 20 cells
#   DATASET=Sheet3 ./run_downstream_TE_esval.sh     # one dataset
#   SPLITS=scaffold ./run_downstream_TE_esval.sh    # one protocol
#   FORCE=1 ./run_downstream_TE_esval.sh            # redo even if the cell exists
#
# Identity (坑 #6 / #14e): this arm shares (dataset, split, seed, lr, epochs) with the
# main arm, so it needs its own identity in FOUR places, all of them wired:
#   * MolTrain run_label  -> `..._esval`        (finetune_moltrain build_run_label)
#   * aggregate key       -> `...|esval`        (finetune_moltrain aggregate key)
#   * run directory       -> `..._esval`        (exp_tag, so it cannot overwrite the
#                                                main arm's cv.data / model_*.pth)
#   * result cell         -> `lipidfm_v5_esval` (collect_neural --model)
# The collector additionally requires `--es-val internal` to match, and disables its
# legacy-dump fallback (that filename carries no run identity and would silently hand
# this arm the MAIN arm's predictions).
#
# 2026-09-19: the first attempt died in `datahub.py:147` -- a dead buffer sized from
# fold 0's two halves (620) while the writes use global row indices (up to 767), which
# only holds when the folds partition the data. Fixed in the vendored tree; pristine
# copy + rationale in `failed/datahub_nfolds_backup_20260919/`.
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"

V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
DATASETS="${DATASETS:-${DATASET:-Sheet2 Sheet3}}"
SPLITS="${SPLITS:-scaffold butina}"
SEEDS="${SEEDS:-42 123 456 789 1011}"
GPUS="${GPUS:-0 1 2}"
LOGDIR="${LOGDIR:-logs/downstream_TE_esval}"
mkdir -p "$LOGDIR"

# Identical to run_downstream_TE.sh. If these ever diverge the two arms stop being a
# controlled comparison and the raw gap measures the hyperparameters instead.
EPOCHS=40; LR=5e-5; ES=8

export LIPIDFM_CHECKPOINT="$V5CKPT"
TAG=esval
ARM_LABEL=lipidfm_v5_esval

# --- idempotence guard (same pattern as run_downstream_TE.sh) --------------------
# Guard on the COLLECTED cell, not the run directory: MolTrain's save_path has no
# timestamp, so the run directory is the thing that gets clobbered and cannot serve
# as evidence a job finished.
RESULTS_DIR="5_outputs/downstream_results"
is_done() {  # $1=dataset $2=split $3=seed
  [ -s "$RESULTS_DIR/$1|${ARM_LABEL}|$2|seed$3|k5.json" ]
}

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
echo "=== $NJOBS jobs, model=$TAG (--es-val internal), datasets=[$DATASETS], splits=[$SPLITS], seeds=[$SEEDS] ==="
echo "=== gpus=[$GPUS], $(date '+%F %T') ==="
if [ "$NJOBS" -eq 0 ]; then echo "nothing to do (all result cells present)"; exit 0; fi
printf '    %s\n' "${JOBS[@]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=== DRY_RUN=1 -- nothing launched. Run without DRY_RUN to execute. ==="
  exit 0
fi

# Deal jobs to GPUs round-robin, one sequential worker per GPU (two MolTrain
# processes on one card OOM -- measured 2026-08-29).
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
          --es-val internal \
          > "$LOGDIR/${NAME}.log" 2>&1 ; then
        # Collect immediately, while this run is unambiguously the newest matching
        # result file. `--es-val internal` + the `_esval` run label are both required:
        # without them the collector would find the MAIN arm's runs (they share every
        # other dimension) and quietly file the wrong numbers under this arm.
        if conda run -n unimol2 --no-capture-output python \
            4_scripts/downstream/collect_neural.py \
            --dataset "$DS" --split "$SPLIT" --seed "$SEED" --model "$ARM_LABEL" \
            --lr "$LR" --epochs "$EPOCHS" --es-val internal --pick-newest \
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
echo "Next: pair this arm against the main one and the classical arms --"
echo "  python 4_scripts/downstream/analyze_eval1.py   # section 1 gains the arm"
echo "The raw gap (default - internal) per (dataset, protocol) IS the 坑 #12"
echo "advantage on the raw R2 scale. Record it in prompt/26-09-18/08-* §3."
