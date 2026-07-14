#!/bin/bash
# Phase 1.5: 更小 lr 扫描
# A5: 200步  lr=2e-6
# A6: 200步  lr=1e-6
# A7: 100步  lr=2e-6
# A8: 100步  lr=1e-6
# 每 50 步保存

set -e
cd /mnt/vepfs/pyten/Programs/code/pi0.6
PYTHON=".venv/bin/python"
SFT_CKPT="checkpoints/pi0_libero/recap_sft_baseline/29999"
DATA="data/rollouts/labeled_episodes_v5.pkl"

run_exp() {
    local name=$1 steps=$2 lr=$3
    echo "[$(date '+%H:%M:%S')] Starting $name: steps=$steps lr=$lr"
    nohup $PYTHON -u scripts/recap/train_arfm.py         --sft-ckpt "$SFT_CKPT"         --labeled-data "$DATA"         --exp-name "arfm_phase1_5_${name}"         --num-steps "$steps"         --batch-size 64         --lr "$lr"         --alpha 2.0         --save-interval 50         > logs/train_phase1_5_${name}.log 2>&1
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

run_exp A5 200 2e-6
run_exp A6 200 1e-6
run_exp A7 100 2e-6
run_exp A8 100 1e-6

echo "Phase 1.5 training done."
