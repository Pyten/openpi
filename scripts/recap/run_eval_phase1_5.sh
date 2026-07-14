#!/bin/bash
# Usage: bash run_eval_phase1_5.sh [num_gpus=1]
set -e
cd /mnt/vepfs/pyten/Programs/code/pi0.6
PYTHON=".venv/bin/python"
SFT="checkpoints/pi0_libero/recap_sft_baseline/29999"
NUM_GPUS="${1:-1}"

echo "Eval Phase1.5 with NUM_GPUS=${NUM_GPUS}"

for exp in A5 A6 A7 A8; do
    echo "[$(date +%H:%M:%S)] Evaluating arfm_phase1_5_${exp} (${NUM_GPUS} GPU(s))..."
    if [ "${NUM_GPUS}" = "1" ]; then
        CUDA_VISIBLE_DEVICES=0 $PYTHON -u scripts/recap/eval_phase1.py \
            --exp-dir checkpoints/pi0_libero/arfm_phase1_5_${exp} \
            --sft-ckpt $SFT \
            --episodes-per-task 5 \
            > logs/eval_phase1_5_${exp}.log 2>&1
    else
        $PYTHON -u scripts/recap/eval_phase1.py \
            --exp-dir checkpoints/pi0_libero/arfm_phase1_5_${exp} \
            --sft-ckpt $SFT \
            --episodes-per-task 5 \
            --num-gpus ${NUM_GPUS} \
            > logs/eval_phase1_5_${exp}.log 2>&1
    fi
    echo "[$(date +%H:%M:%S)] Done ${exp}"
done
echo "Phase1.5 evals done."
