#!/usr/bin/env bash
# PEFT 消融队列：串行跑剩余 arm。每档 5 折约 90 分钟。
#
# 为什么串行而不是并行：单卡，且 MolTrain 无折级断点续训（坑 #8）——
# 并行只会让两者都变慢，还增加 GPU 掉线时的损失。
# 为什么一次只排 4 档：GPU 已掉线 3 次（其中一次空闲时发生）。每档是原子的，
# 一次排太多会在掉线时浪费已完成之外的全部时间。
set -uo pipefail
CT="/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training"
cd "$CT"

QUEUE=(
  "C butina 1337"      # butina 第二种子
  "C butina 2024"
  "D scaffold 1337"    # D 是唯一有可能追平 A 的 arm，补齐 15 折
  "D scaffold 2024"
)

for item in "${QUEUE[@]}"; do
  read -r cfg split seed <<< "$item"
  echo "########## queue: $cfg $split seed=$seed  $(date '+%F %T')"
  bash run_peft_eval.sh "$cfg" "$split" "$seed"
  echo "########## finished: $cfg $split seed=$seed  $(date '+%F %T')"
done
echo "########## QUEUE COMPLETE $(date '+%F %T')"
