#!/bin/bash
# Usage: bash run_eval_phase1.sh [num_gpus=1]
# Example: bash run_eval_phase1.sh 4   (4-GPU parallel)
set -e
cd /mnt/vepfs/pyten/Programs/code/pi0.6
PYTHON=".venv/bin/python"
SFT="checkpoints/pi0_libero/recap_sft_baseline/29999"
NUM_GPUS="${1:-1}"

echo "Eval with NUM_GPUS=${NUM_GPUS}"

for exp in A1 A2 A3 A4; do
    echo "[$(date +%H:%M:%S)] Evaluating arfm_phase1_${exp} (${NUM_GPUS} GPU(s))..."
    if [ "${NUM_GPUS}" = "1" ]; then
        CUDA_VISIBLE_DEVICES=0 $PYTHON -u scripts/recap/eval_phase1.py \
            --exp-dir checkpoints/pi0_libero/arfm_phase1_${exp} \
            --sft-ckpt $SFT \
            --episodes-per-task 5 \
            > logs/eval_phase1_${exp}.log 2>&1
    else
        $PYTHON -u scripts/recap/eval_phase1.py \
            --exp-dir checkpoints/pi0_libero/arfm_phase1_${exp} \
            --sft-ckpt $SFT \
            --episodes-per-task 5 \
            --num-gpus ${NUM_GPUS} \
            > logs/eval_phase1_${exp}.log 2>&1
    fi
    echo "[$(date +%H:%M:%S)] Done ${exp}"
done
echo "All evals done."
