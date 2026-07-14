#!/bin/bash
# Phase 2: 精细扫描
# B1: 200步  lr=5e-7  (比A6更小lr，看峰值是否后移)
# B2: 200步  lr=3e-7  (更极小lr)
# B3: 200步  lr=1e-6  (重跑A6，save-interval=25，精确定位峰值)
# B4: 200步  lr=2e-7  (探索极限)
# 每 25 步保存（更细粒度）

set -e
cd /mnt/vepfs/pyten/Programs/code/pi0.6
PYTHON=".venv/bin/python"
SFT_CKPT="checkpoints/pi0_libero/recap_sft_baseline/29999"
DATA="data/rollouts/labeled_episodes_v5.pkl"

run_exp() {
    local name=$1 steps=$2 lr=$3
    echo "[$(date '+%H:%M:%S')] Starting $name: steps=$steps lr=$lr"
    $PYTHON -u scripts/recap/train_arfm.py \
        --sft-ckpt "$SFT_CKPT" \
        --labeled-data "$DATA" \
        --exp-name "arfm_phase2_${name}" \
        --num-steps "$steps" \
        --batch-size 64 \
        --lr "$lr" \
        --alpha 2.0 \
        --save-interval 25 \
        > logs/train_phase2_${name}.log 2>&1
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

run_exp B1 200 5e-7
run_exp B2 200 3e-7
run_exp B3 200 1e-6
run_exp B4 200 2e-7

echo "Phase 2 training done."
