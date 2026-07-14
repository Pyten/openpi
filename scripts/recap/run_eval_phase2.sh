#!/bin/bash
# Phase 2 评测：B1/B2/B3/B4
# Usage: bash scripts/recap/run_eval_phase2.sh [NUM_GPUS]

set -e
cd /mnt/vepfs/pyten/Programs/code/pi0.6
NUM_GPUS=${1:-4}
PYTHON=".venv/bin/python"
SFT_CKPT="checkpoints/pi0_libero/recap_sft_baseline/29999"

echo "Eval Phase2 with NUM_GPUS=${NUM_GPUS}"

for exp in B1 B2 B3 B4; do
    echo "[$(date '+%H:%M:%S')] Evaluating arfm_phase2_${exp} (${NUM_GPUS} GPU(s))..."
    EXP_DIR="checkpoints/pi0_libero/arfm_phase2_${exp}"
    if [ ! -d "$EXP_DIR" ]; then
        echo "  [SKIP] $EXP_DIR not found"
        continue
    fi
    $PYTHON scripts/recap/eval_phase1.py \
        --exp-dir "$EXP_DIR" \
        --sft-ckpt "$SFT_CKPT" \
        --num-gpus "$NUM_GPUS" \
        > logs/eval_phase2_${exp}.log 2>&1
    echo "[$(date '+%H:%M:%S')] Done ${exp}"
done

echo "Phase 2 eval done."
