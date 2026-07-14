#!/bin/bash
# Phase 1: ARFM 过拟合甜蜜点探索
# A1: 500步  lr=1e-5
# A2: 1000步 lr=1e-5
# A3: 2000步 lr=1e-5
# A4: 2000步 lr=5e-6
# 每 200 步保存 checkpoint
#
# Usage: bash scripts/recap/run_phase1.sh [A1|A2|A3|A4|all]
#   default: all (sequential)

set -e
cd /mnt/vepfs/pyten/Programs/code/pi0.6

PYTHON=".venv/bin/python"
SFT_CKPT="checkpoints/pi0_libero/recap_sft_baseline/29999"
DATA="data/rollouts/labeled_episodes_v5.pkl"
CKPT_BASE="checkpoints"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local name=$1 steps=$2 lr=$3
    echo "[$(date '+%H:%M:%S')] Starting $name: steps=$steps lr=$lr"
    nohup $PYTHON -u scripts/recap/train_arfm.py \
        --sft-ckpt "$SFT_CKPT" \
        --labeled-data "$DATA" \
        --exp-name "arfm_phase1_${name}" \
        --num-steps "$steps" \
        --batch-size 64 \
        --lr "$lr" \
        --alpha 2.0 \
        --save-interval 200 \
        > "$LOG_DIR/train_phase1_${name}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

TARGET="${1:-all}"

case "$TARGET" in
    A1) run_exp A1 500  1e-5 ;;
    A2) run_exp A2 1000 1e-5 ;;
    A3) run_exp A3 2000 1e-5 ;;
    A4) run_exp A4 2000 5e-6 ;;
    all)
        run_exp A1 500  1e-5
        run_exp A2 1000 1e-5
        run_exp A3 2000 1e-5
        run_exp A4 2000 5e-6
        ;;
    *)
        echo "Usage: $0 [A1|A2|A3|A4|all]"
        exit 1 ;;
esac

echo "Phase 1 training done."
