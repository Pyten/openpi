#!/bin/bash
# Phase 3: 锁定 lr=5e-7, 精细扫描更早步数
# C1: 100步  lr=5e-7  save-interval=10  (探索 step=10,20,30... 的真实峰值)
# C2: 100步  lr=8e-7  save-interval=10  (5e-7 和 1e-6 之间)
# C3: 100步  lr=3e-7  save-interval=10  (再次验证 3e-7)
# C4: 50步   lr=5e-7  save-interval=5   (超早期探索)

set -e
cd /mnt/vepfs/pyten/Programs/code/pi0.6
PYTHON=".venv/bin/python"
SFT_CKPT="checkpoints/pi0_libero/recap_sft_baseline/29999"
DATA="data/rollouts/labeled_episodes_v5.pkl"

run_exp() {
    local name=$1 steps=$2 lr=$3 interval=$4
    echo "[$(date '+%H:%M:%S')] Starting $name: steps=$steps lr=$lr interval=$interval"
    $PYTHON -u scripts/recap/train_arfm.py \
        --sft-ckpt "$SFT_CKPT" \
        --labeled-data "$DATA" \
        --exp-name "arfm_phase3_${name}" \
        --num-steps "$steps" \
        --batch-size 64 \
        --lr "$lr" \
        --alpha 2.0 \
        --save-interval "$interval" \
        > logs/train_phase3_${name}.log 2>&1
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

run_exp C1 100 5e-7 10
run_exp C2 100 8e-7 10
run_exp C3 100 3e-7 10
run_exp C4 50  5e-7 5

echo "Phase 3 training done."
