#!/usr/bin/env bash
# 微调侧实验：v5 backbone + 冻结前 6 层（Sheet2 scaffold s1337）
# 对照：v5 不冻结 = 0.673（历史结果）
set -euo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"
V5CKPT="$CT/2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt"

wait_gpu_free() {
  echo "===== waiting for GPU to be free $(date '+%T')"
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -lt 1000 ]; then echo "GPU free ($used MiB)"; return; fi
    sleep 120
  done
}
wait_gpu_free

echo "===== [finetune_v5_freeze6_Sheet2_scaf_s1337] $(date '+%F %T')"
env LIPIDFM_CHECKPOINT="$V5CKPT" conda run -n unimol2 --no-capture-output python \
  4_scripts/finetune_moltrain.py --dataset Sheet2 --split scaffold --seed 1337 \
  --lr 5e-5 --epochs 40 --early-stopping 8 --kfold 5 --freeze-layers 6 \
  > logs/finetune_v5_freeze6_Sheet2_scaf_s1337.log 2>&1 || echo "!!!! FAILED"
echo "===== done $(date '+%F %T')"
