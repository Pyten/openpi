#!/bin/bash
# Phase 3 评测
set -e
cd /mnt/vepfs/pyten/Programs/code/pi0.6
NUM_GPUS=${1:-4}
PYTHON=".venv/bin/python"
SFT_CKPT="checkpoints/pi0_libero/recap_sft_baseline/29999"

echo "Eval Phase3 with NUM_GPUS=${NUM_GPUS}"

for exp in C1 C2 C3 C4; do
    echo "[$(date '+%H:%M:%S')] Evaluating arfm_phase3_${exp} (${NUM_GPUS} GPU(s))..."
    EXP_DIR="checkpoints/pi0_libero/arfm_phase3_${exp}"
    if [ ! -d "$EXP_DIR" ]; then
        echo "  [SKIP] $EXP_DIR not found"
        continue
    fi
    $PYTHON scripts/recap/eval_phase1.py \
        --exp-dir "$EXP_DIR" \
        --sft-ckpt "$SFT_CKPT" \
        --num-gpus "$NUM_GPUS" \
        > logs/eval_phase3_${exp}.log 2>&1
    echo "[$(date '+%H:%M:%S')] Done ${exp}"
done

echo "Phase 3 eval done."
