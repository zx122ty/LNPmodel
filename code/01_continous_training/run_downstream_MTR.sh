#!/usr/bin/env bash
# run_downstream_MTR.sh — evaluation 3 (MTR): 83 molecules x 16 multi-modal assays.
#
# Protocol (user decision 2026-09-18, plan 26-09-16 §5.4)
# -----------------------------------------------------
#   * `random` is the MAIN split. MTR has 83 molecules and 16 targets; the splitter
#     choice cannot rescue the statistical power, and that reason is independent of
#     any splitter.
#   * `scaffold` is an explicitly WARNED appendix. Its family-concentration ratio is
#     1.93 (largest scaffold 32 vs mean fold 16.6), past the 1.0 threshold where a
#     fold is filled by a single chemical family and a constant predictor does as
#     well -- measured R2 -3.93. It is run because a warned number with its warning
#     attached is more useful than an absent one, and because 坑 #13's discriminant
#     deserves a second dataset. Do not quote it without the warning.
#   * `butina` is NOT offered at any cutoff (split_files.SPLITS_BY_DATASET).
#
# Arms (user decision 2026-09-18)
# ------------------------------
#   neural, default        `lipidfm_v5`         early stopping on the TEST fold (坑 #12)
#   neural, control        `lipidfm_v5_esval`   early stopping on a train-internal 15%
#                                               holdout -- the SAME protocol as the
#                                               classical arms, so its raw R2 can sit
#                                               beside theirs and the raw gap IS the
#                                               坑 #12 advantage, measured directly
#   classical              `ecfp4_rf`, `ecfp4_xgb`
#
# `mlp_morgan` is deliberately absent: it is single-output by construction
# (`baselines.make_mlp_morgan`), and 16 copies of it is a different experiment, not a
# multi-output baseline. RF/XGB loop `for j in range(Y.shape[1])` natively.
#
# Cost
# ----
#   neural: 2 arms x 2 splits x 5 seeds = 20 cells x 5 folds = 100 fold-jobs, 40 epochs.
#           MTR molecules are small (83 rows, ~4 batches/fold), so this is far cheaper
#           per fold than Sheet2's 26.4 min -- measured per-cell wall time is recorded
#           in the startup record. Round-robin over 3 GPUs, one MolTrain per card
#           (two on one card OOM, measured 2026-08-29).
#   classical: 2 models x 2 splits x 5 seeds = 20 cells on CPU (RF ~36 s, XGB ~110 s
#           wall / ~110 min CPU per cell). Run AFTER the neural phase so the two do
#           not contend -- 坑 #14d: a contended wall time is not a portable number.
#
# Usage
#   DRY_RUN=1 ./run_downstream_MTR.sh              # print the job list, launch nothing
#   ./run_downstream_MTR.sh                        # neural then classical
#   PHASE=neural ./run_downstream_MTR.sh           # one phase
#   SPLITS=random ./run_downstream_MTR.sh          # main protocol only
#   FORCE=1 ./run_downstream_MTR.sh                # redo even if the cell exists
#
# Identity (坑 #6 / #14e): the two neural arms share (dataset, split, seed, lr, epochs)
# and differ only in which fold early stopping watched, so the control arm carries its
# own identity in FOUR places, all wired in finetune_moltrain / collect_neural:
#   run_label `..._esval` | aggregate key `...|esval` | run dir `..._esval` | cell `..._esval`
# The collector also requires `--es-val internal` to match, and disables its legacy-dump
# fallback (that filename carries no run identity and would hand this arm the main
# arm's predictions).
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"

V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
DATASET=MTR
# `random` is the main protocol; `scaffold` is the warned appendix. Order matters for
# nothing but the log, and random is listed first so it is the one a partial run gets.
SPLITS="${SPLITS:-random scaffold}"
SEEDS="${SEEDS:-42 123 456 789 1011}"
GPUS="${GPUS:-0 1 2}"
PHASE="${PHASE:-all}"
LOGDIR="${LOGDIR:-logs/downstream_MTR}"
mkdir -p "$LOGDIR"

# The project's standard downstream budget, unchanged so MTR is comparable to the
# other two evaluations: lr 5e-5, 40 epochs, early-stopping 8, kfold 5, batch 16.
EPOCHS=40; LR=5e-5; ES=8

export LIPIDFM_CHECKPOINT="$V5CKPT"

RESULTS_DIR="5_outputs/downstream_results"
NEURAL_ARMS="lipidfm_v5 lipidfm_v5_esval"
CLASSICAL_MODELS="ecfp4_rf ecfp4_xgb"

# --- idempotence guard ------------------------------------------------------------
# On the COLLECTED cell, never on the run directory: MolTrain's save_path has no
# timestamp, so the run directory is exactly the thing a re-run clobbers and cannot
# serve as evidence that a job finished (05-pKa评估结果.md §9.1 gap #1).
is_done() {  # $1=model-label $2=split $3=seed
  [ -s "$RESULTS_DIR/${DATASET}|$1|$2|seed$3|k5.json" ]
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=== DRY_RUN=1 -- job list only ==="
  for ARM in $NEURAL_ARMS; do
    for S in $SPLITS; do for SD in $SEEDS; do
      st=$([ -s "$RESULTS_DIR/${DATASET}|${ARM}|${S}|seed${SD}|k5.json" ] && echo EXISTS || echo todo)
      echo "  neural    $ARM/$S/seed$SD   $st"
    done; done
  done
  for M in $CLASSICAL_MODELS; do
    for S in $SPLITS; do for SD in $SEEDS; do
      st=$([ -s "$RESULTS_DIR/${DATASET}|${M}|${S}|seed${SD}|k5.json" ] && echo EXISTS || echo todo)
      echo "  classical $M/$S/seed$SD   $st"
    done; done
  done
  exit 0
fi

# ==================================================================================
# Phase 1: neural arms (GPU)
# ==================================================================================
if [ "$PHASE" = "all" ] || [ "$PHASE" = "neural" ]; then
  JOBS=()
  for ARM in $NEURAL_ARMS; do
    for S in $SPLITS; do
      for SD in $SEEDS; do
        if [ "${FORCE:-0}" != "1" ] && is_done "$ARM" "$S" "$SD"; then
          echo "=== SKIP $DATASET/$ARM/$S/seed$SD -- result cell already exists (FORCE=1 to redo) ==="
          continue
        fi
        JOBS+=("$ARM:$S:$SD")
      done
    done
  done
  NJOBS=${#JOBS[@]}
  echo "=== [neural] $NJOBS cells, splits=[$SPLITS], seeds=[$SEEDS], gpus=[$GPUS], $(date '+%F %T') ==="
  if [ "$NJOBS" -gt 0 ]; then
    printf '    %s\n' "${JOBS[@]}"
    NGPU=$(echo $GPUS | wc -w)
    i=0
    for G in $GPUS; do
      (
        for j in "${!JOBS[@]}"; do
          [ $(( j % NGPU )) -ne $i ] && continue
          IFS=: read -r ARM SPLIT SEED <<< "${JOBS[$j]}"
          ESFLAG=""
          [ "$ARM" = "lipidfm_v5_esval" ] && ESFLAG="--es-val internal"
          NAME="finetune_${ARM}_${DATASET}_${SPLIT}_s${SEED}"
          echo "===== [$NAME] gpu$G $(date '+%F %T')"
          # --butina-cutoff is passed for symmetry with the other queues even though
          # MTR has no butina split: effective_butina_cutoff only consults it for a
          # butina run, and the frozen random/scaffold folds never read it.
          if env CUDA_VISIBLE_DEVICES="$G" conda run -n unimol2 --no-capture-output python \
              4_scripts/finetune_moltrain.py \
              --dataset "$DATASET" --split "$SPLIT" --butina-cutoff 0.8 \
              --seed "$SEED" --lr "$LR" --epochs "$EPOCHS" --early-stopping "$ES" --kfold 5 \
              $ESFLAG \
              > "$LOGDIR/${NAME}.log" 2>&1 ; then
            # Collect immediately, while this run is unambiguously the newest matching
            # result file. `--es-val internal` + the `_esval` run label are both
            # required for the control arm: without them the collector finds the MAIN
            # arm's runs (they share every other dimension) and files the wrong numbers.
            if conda run -n unimol2 --no-capture-output python \
                4_scripts/downstream/collect_neural.py \
                --dataset "$DATASET" --split "$SPLIT" --seed "$SEED" --model "$ARM" \
                --lr "$LR" --epochs "$EPOCHS" \
                $([ "$ARM" = "lipidfm_v5_esval" ] && echo "--es-val internal") \
                --pick-newest >> "$LOGDIR/${NAME}.log" 2>&1 ; then
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
    echo "=== [neural] done $(date '+%F %T') ==="
  else
    echo "=== [neural] nothing to do (all result cells present) ==="
  fi
fi

# ==================================================================================
# Phase 2: classical arms (CPU) -- AFTER the neural phase, so no GPU queue is
# contending for cores while these are timed (坑 #14d).
# ==================================================================================
if [ "$PHASE" = "all" ] || [ "$PHASE" = "classical" ]; then
  echo "=== [classical] $CLASSICAL_MODELS over splits=[$SPLITS], seeds=[$SEEDS], $(date '+%F %T') ==="
  for M in $CLASSICAL_MODELS; do
    for S in $SPLITS; do
      # Skip whole (model, split) groups only when EVERY seed is already collected;
      # the runner itself has no per-seed guard, so re-running a group would redo
      # cells that exist. Per-seed skipping is done by the runner being cheap enough
      # to redo -- but a partial group would silently re-run collected seeds, so the
      # group is skipped only when it is complete.
      ALLDONE=1
      for SD in $SEEDS; do
        is_done "$M" "$S" "$SD" || ALLDONE=0
      done
      if [ "${FORCE:-0}" != "1" ] && [ "$ALLDONE" = "1" ]; then
        echo "=== SKIP $DATASET/$M/$S -- all seeds collected (FORCE=1 to redo) ==="
        continue
      fi
      NAME="baseline_${M}_${DATASET}_${S}"
      echo "===== [$NAME] cpu $(date '+%F %T')"
      if conda run -n unimol2 --no-capture-output python \
          4_scripts/downstream/run_downstream_baselines.py \
          --dataset "$DATASET" --split "$S" --models "$M" --seeds $SEEDS \
          > "$LOGDIR/${NAME}.log" 2>&1 ; then
        echo "===== [$NAME] done $(date '+%F %T')"
      else
        echo "!!!! [$NAME] FAILED $(date '+%T') (see $LOGDIR/${NAME}.log)"
      fi
    done
  done
  echo "=== [classical] done $(date '+%F %T') ==="
fi

echo "ALL DONE $(date '+%F %T')"
echo
echo "Next: the per-assay table (16 assays, BH-FDR across them, pooled screening)"
echo "  python 4_scripts/downstream/report_table.py --dataset MTR --split random \\"
echo "      --models lipidfm_v5 lipidfm_v5_esval ecfp4_rf ecfp4_xgb"
echo "  # the warned appendix:"
echo "  python 4_scripts/downstream/report_table.py --dataset MTR --split scaffold \\"
echo "      --models lipidfm_v5 lipidfm_v5_esval ecfp4_rf ecfp4_xgb"
