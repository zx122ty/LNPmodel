#!/usr/bin/env bash
# run_add_lumi_stock.sh — 2026-09-22: add `stock_84m` and LUMI-lab to the evaluation
# pipeline. Startup record: prompt/26-09-22/01-加入Lumi-lab与stock_84m-启动记录.md
#
# Why these four phases and this order
# ------------------------------------
#   Phase 1  stock_84m on Sheet2/scaffold      -- zero new code; answers "did the
#   Phase 2  stock_84m on Sheet3/butina           continued pretraining (v5) earn
#                                                 anything over its own base?"
#   Phase 3  lumi_lab_esval on Sheet2 x 2 splits -- lifts the only arm below the
#                                                 15-fold floor to 25 folds
#   Phase 4  lumi_lab_esval + lumi_lab on Sheet3/butina -- the DISCRIMINATING cell
#                                                 for 坑 #12 on LUMI-lab (Sheet2's
#                                                 +0.023/+0.027 is inside the
#                                                 0.0335 run-to-run noise)
#
# Two separate calls for stock, NOT one with DATASETS="Sheet2 Sheet3"
# SPLITS="scaffold butina": that variable pair is a cartesian product (4 combos,
# 100 folds) and would also run Sheet2/butina and Sheet3/scaffold, neither of which
# is in scope (and Sheet3/scaffold is not usable at all -- 坑 #13, family
# concentration 1.14 > 1.0).
#
# Phases are SERIAL. LUMI-lab and MolTrain concurrently would land on the same card
# (two MolTrain on one card OOM, measured 2026-08-29) and would make every wall time
# unportable (坑 #14d: contention moves wall time 80x and nothing else).
#
# Idempotent: every phase's guard tests the COLLECTED cell, never the run directory
# (MolTrain's save_path has no timestamp -- 坑 #6/#14b). Re-running this script
# skips everything already collected. FORCE=1 redoes.
#
# Usage:
#   DRY_RUN=1 ./run_add_lumi_stock.sh          # print the plan, launch nothing
#   nohup ./run_add_lumi_stock.sh > logs/add_lumi_stock.log 2>&1 &
set -uo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
LUMI="/home/data/zhengxiao/disk1/my_project/lnpModel/other_model/LUMI-lab/deploy"
cd "$CT"
mkdir -p logs

# Sheet2 gets GPU 0 first: at 2026-09-22 09:53 GPU 0 had 4.7 GB used (80 GB free)
# while GPUs 1/2 had 40 GB used each by another user's nafnet/unet jobs.
GPUS="${GPUS:-0 1 2}"
LUMI_GPUS="${LUMI_GPUS:-0 1}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=== PHASE 1: stock_84m / Sheet2 / scaffold ==="
  DRY_RUN=1 MODELS=stock DATASETS=Sheet2 SPLITS=scaffold GPUS="$GPUS" \
      ./run_downstream_TE.sh
  echo "=== PHASE 2: stock_84m / Sheet3 / butina ==="
  DRY_RUN=1 MODELS=stock DATASETS=Sheet3 SPLITS=butina GPUS="$GPUS" \
      ./run_downstream_TE.sh
  echo "=== PHASE 3: lumi_lab_esval / Sheet2 / scaffold+butina / 5 seeds ==="
  (cd "$LUMI" && CUDA_VISIBLE_DEVICES=0 python lumi_pipeline_adapter.py --dataset Sheet2 \
      --split scaffold --seeds 42 123 456 789 1011 --arms lumi_lab_esval --dry-run 2>&1 | grep -E "^===|conformers|fine-tunes")
  (cd "$LUMI" && CUDA_VISIBLE_DEVICES=0 python lumi_pipeline_adapter.py --dataset Sheet2 \
      --split butina --seeds 42 123 456 789 1011 --arms lumi_lab_esval --dry-run 2>&1 | grep -E "^===|conformers|fine-tunes")
  echo "=== PHASE 4: lumi_lab_esval + lumi_lab / Sheet3 / butina / 5 seeds ==="
  (cd "$LUMI" && CUDA_VISIBLE_DEVICES=0 python lumi_pipeline_adapter.py --dataset Sheet3 \
      --split butina --seeds 42 123 456 789 1011 --arms lumi_lab_esval lumi_lab --dry-run 2>&1 | grep -E "^===|conformers|fine-tunes")
  echo "=== DRY_RUN=1 -- nothing launched ==="
  exit 0
fi

echo "########## PHASE 1: stock_84m / Sheet2 / scaffold $(date '+%F %T') ##########"
MODELS=stock DATASETS=Sheet2 SPLITS=scaffold GPUS="$GPUS" \
    ./run_downstream_TE.sh 2>&1 | tee -a logs/add_lumi_stock.log

echo "########## PHASE 2: stock_84m / Sheet3 / butina $(date '+%F %T') ##########"
MODELS=stock DATASETS=Sheet3 SPLITS=butina GPUS="$GPUS" \
    ./run_downstream_TE.sh 2>&1 | tee -a logs/add_lumi_stock.log

cd "$LUMI"
mkdir -p logs
echo "########## PHASE 3a: lumi_lab_esval / Sheet2 / scaffold $(date '+%F %T') ##########"
CUDA_VISIBLE_DEVICES="${LUMI_GPUS%% *}" python lumi_pipeline_adapter.py --dataset Sheet2 \
    --split scaffold --seeds 42 123 456 789 1011 --arms lumi_lab_esval \
    2>&1 | tee -a "$CT/logs/add_lumi_stock.log"

echo "########## PHASE 3b: lumi_lab_esval / Sheet2 / butina $(date '+%F %T') ##########"
CUDA_VISIBLE_DEVICES="${LUMI_GPUS##* }" python lumi_pipeline_adapter.py --dataset Sheet2 \
    --split butina --seeds 42 123 456 789 1011 --arms lumi_lab_esval \
    2>&1 | tee -a "$CT/logs/add_lumi_stock.log"

echo "########## PHASE 4: lumi_lab_esval + lumi_lab / Sheet3 / butina $(date '+%F %T') ##########"
CUDA_VISIBLE_DEVICES="${LUMI_GPUS%% *}" python lumi_pipeline_adapter.py --dataset Sheet3 \
    --split butina --seeds 42 123 456 789 1011 --arms lumi_lab_esval lumi_lab \
    2>&1 | tee -a "$CT/logs/add_lumi_stock.log"

echo "########## ALL PHASES DONE $(date '+%F %T') ##########"
