#!/usr/bin/env bash
# run_downstream_TE_baselines.sh — evaluation 1 (TE) classical-arm queue.
#
# The CPU counterpart of run_downstream_TE.sh: the three classical baselines on
# the SAME frozen folds the neural queue reads (5_outputs/downstream_splits/), so
# every arm can be paired fold-for-fold by report_table.py.
#
# Why this is a separate script and not part of run_downstream_TE.sh:
#   * the neural queue is GPU-bound (3 jobs at a time, ~10 h wall); this one is
#     CPU-bound and the two would fight for cores -- and 坑 #14d is precisely the
#     record of what that fight does to a number (wall clock off by 80x).
#   * the arms are launched at different times, so the neural queue should not
#     block on CPU work it does not need.
#
# Ordering: run this AFTER the neural queue finishes, or at least accept that the
# elapsed_sec it records will be contended wall clock. r2_per_fold and the OOF
# arrays are unaffected by contention (measured, 坑 #14d) -- only elapsed_sec is.
# The honest way to read elapsed_sec here is `cost_probe.py`, which reports CPU
# time; the numbers below are budget, not measurement.
#
# Seeds: the frozen store carries 7 for scaffold/butina (plan grid {42,123,456,
# 789,1011} + historical {1337,2024}). All 7 are run here -- the CPU arms are
# cheap (see budget) and this keeps the store's convention that every evaluation-1
# protocol covers the same seeds, so any later analysis can slice either the
# 25-fold plan grid or the full 35 folds without a re-run.
# ⚠️ The HEADLINE is the 25-fold plan grid (so it pairs fold-for-fold with pKa and
# MTR); pass `--seeds 42 123 456 789 1011` to report_table.py. The extra two seeds
# are not part of the headline and must not be mixed into it.
#
# CPU budget (measured, cost_probe.py; see prompt/26-09-18/05-* §9.4) -- cpu-s/fold:
#   Sheet2/scaffold  rf 11.3   xgb 416.3   mlp 370.5
#   Sheet2/butina    rf 11.1   xgb 209.2   mlp 390.1
#   Sheet3/scaffold  rf  6.1   xgb 183.8   mlp 303.9
#   Sheet3/butina    rf  5.7   xgb 182.8   mlp 572.1
# => 7 seeds x 5 folds = 35 folds per (dataset, protocol):
#    Sheet2: (11.3+416.3+370.5) + (11.1+209.2+390.1) = 798.1 + 610.4 = 1408.5 cpu-s/fold-set
#    Sheet3: (6.1+183.8+303.9) + (5.7+182.8+572.1)  = 493.8 + 760.6 = 1254.4
#    total ~2663 cpu-s per fold x 35 folds ~= 25.9 cpu-h.
# On a 64-core box that is under 30 min of wall clock if it has the machine to
# itself -- but xgboost/sklearn default to n_jobs=-1 (all cores), so with the
# neural queue running expect the wall clock to be several times that while the
# CPU time stays put.
#
# Usage:
#   DRY_RUN=1 ./run_downstream_TE_baselines.sh      # print the job list
#   ./run_downstream_TE_baselines.sh                # both datasets, 3 arms
#   DATASET=Sheet3 ./run_downstream_TE_baselines.sh # one dataset
#   MODELS=ecfp4_rf ./run_downstream_TE_baselines.sh
#   FORCE=1 ./run_downstream_TE_baselines.sh        # redo existing cells
#
# 2026-09-18: the Sheet2/Sheet3 butina folds were rebuilt (坑 #14, wrong distance
# triangle). Every butina number from here on comes from the corrected frozen
# folds; the old cells are quarantined under
# failed/butina_distance_order_bug_20260918/. scaffold was never affected.
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"

# DATASET (singular) accepted as an alias -- same reason as run_downstream_TE.sh:
# every other script in this project says `--dataset`, and a variable that silently
# does nothing is worse than one that errors.
DATASETS="${DATASETS:-${DATASET:-Sheet2 Sheet3}}"
SPLITS="${SPLITS:-scaffold butina}"
SEEDS="${SEEDS:-42 123 456 789 1011 1337 2024}"
MODELS="${MODELS:-ecfp4_rf ecfp4_xgb mlp_morgan}"
LOGDIR="${LOGDIR:-logs/downstream_TE}"
mkdir -p "$LOGDIR"

# --- idempotence guard (same pattern as run_downstream_pKa.sh / run_downstream_TE.sh) ---
# The runner keys each cell by (dataset, model, split, seed) and overwrites in
# place, so a re-launch would silently replace a finished arm's numbers. The cell
# filename is built exactly the way run_matrix writes it -- if this pattern drifts
# from run_matrix's, the guard silently never fires, which is the failure it exists
# to prevent. (That is not hypothetical: collect_neural's model dimension had
# exactly this shape of drift, 06-* §4.6.3.)
is_done() {  # $1=dataset $2=split $3=seed $4=model
  [ -s "5_outputs/downstream_results/$1|$4|$2|seed$3|k5.json" ]
}

JOBS=()
for DS in $DATASETS; do
  for S in $SPLITS; do
    for SD in $SEEDS; do
      for M in $MODELS; do
        if [ "${FORCE:-0}" != "1" ] && is_done "$DS" "$S" "$SD" "$M"; then
          echo "=== SKIP $DS/$S/seed$SD/$M -- result cell already exists (FORCE=1 to redo) ==="
          continue
        fi
        JOBS+=("$DS:$S:$SD:$M")
      done
    done
  done
done
NJOBS=${#JOBS[@]}
echo "=== $NJOBS jobs, datasets=[$DATASETS], splits=[$SPLITS], seeds=[$SEEDS], models=[$MODELS] ==="
echo "=== $(date '+%F %T') ==="
if [ "$NJOBS" -eq 0 ]; then echo "nothing to do (all result cells present)"; exit 0; fi
printf '    %s\n' "${JOBS[@]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=== DRY_RUN=1 -- nothing launched. Run without DRY_RUN to execute. ==="
  exit 0
fi

# Grouped by (dataset, split, model) rather than one process per cell: the runner
# computes the Morgan fingerprints ONCE per invocation (~30 s of RDKit on 1920
# molecules) and loops the seeds internally, so grouping turns 7 invocations into
# 1. It also means a crash loses one arm's seeds rather than the whole queue.
for DS in $DATASETS; do
  for S in $SPLITS; do
    for M in $MODELS; do
      PENDING=()
      for SD in $SEEDS; do
        [ "${FORCE:-0}" != "1" ] && is_done "$DS" "$S" "$SD" "$M" && continue
        PENDING+=("$SD")
      done
      [ ${#PENDING[@]} -eq 0 ] && continue
      NAME="baselines_${M}_${DS}_${S}"
      echo "===== [$NAME] seeds=[${PENDING[*]}] $(date '+%F %T')"
      if python 4_scripts/downstream/run_downstream_baselines.py \
           --dataset "$DS" --split "$S" --models "$M" \
           --seeds "${PENDING[@]}" > "$LOGDIR/${NAME}.log" 2>&1 ; then
        echo "===== [$NAME] done $(date '+%F %T')"
      else
        echo "!!!! [$NAME] FAILED $(date '+%T') (see $LOGDIR/${NAME}.log)"
      fi
    done
  done
done

echo "ALL DONE $(date '+%F %T')"
echo "Next: the paired table per (dataset, protocol). Pin --seeds to the plan grid --"
echo "the store holds 7 seeds but the headline is the 25-fold grid, so that it pairs"
echo "fold-for-fold with pKa and MTR:"
echo "  python 4_scripts/downstream/report_table.py --dataset Sheet2 --split scaffold \\"
echo "    --models lipidfm_v5 ecfp4_rf ecfp4_xgb mlp_morgan --seeds 42 123 456 789 1011"
