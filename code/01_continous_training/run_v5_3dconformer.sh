#!/usr/bin/env bash
# 3D-conformer fine-tune: the paired test of the conformer-pipeline fix.
#
# Baseline to beat: v5 on Sheet2 scaffold seed1337 = fold-mean R2 0.673, run through
# MolTrain's built-in conformer generator, which substitutes a FLAT 2D depiction for
# 81.8% of these molecules (see 4_scripts/conformer_3d.py).
#
# This run changes exactly ONE thing: it feeds ETKDGv3 + MMFF94s 3D conformers
# (100% coverage, verified) through the atoms/coordinates path. Same checkpoint, same
# cached folds, same lr/epochs/early-stopping, same seed -- so the difference is
# attributable to the conformers alone.
#
# Usage: bash run_v5_3dconformer.sh
set -euo pipefail
cd "$(dirname "$0")"

V5CKPT="$PWD/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"
LOG="$PWD/logs/finetune_v5_3dconf_Sheet2_scaf_s1337.log"

echo "checkpoint : $V5CKPT"
echo "log        : $LOG"
echo "conformers : pre-generated ETKDGv3+MMFF94s (5_outputs/conf3d_cache)"
echo

env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
  4_scripts/finetune_moltrain.py \
    --dataset Sheet2 \
    --split scaffold \
    --seed 1337 \
    --lr 5e-5 \
    --epochs 40 \
    --early-stopping 8 \
    --kfold 5 \
    --use-3d-conformers \
  2>&1 | tee "$LOG"
