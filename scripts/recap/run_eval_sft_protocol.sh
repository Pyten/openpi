#!/usr/bin/env bash
set -euo pipefail

cd /mnt/vepfs/pyten/Programs/code/pi0.6

PYTHON=".venv/bin/python"
SFT_CKPT="${SFT_CKPT:-checkpoints/pi0_libero/recap_sft_baseline/29999}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
LOG_DIR="${LOG_DIR:-logs/protocol_v1}"
mkdir -p "$LOG_DIR"

run_seed() {
    local seed="$1"
    local gpu="$2"
    local log="$LOG_DIR/sft_seed${seed}.log"
    echo "[$(date '+%F %T')] seed=$seed gpu=$gpu start" | tee -a "$LOG_DIR/run.log"
    CUDA_VISIBLE_DEVICES="$gpu" \
    MUJOCO_GL=osmesa \
    XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
        "$PYTHON" -u scripts/recap/eval_recap.py \
        --sft-ckpt "$SFT_CKPT" \
        --episodes-per-task "$EPISODES_PER_TASK" \
        --seed "$seed" \
        --eval-sft --no-eval-recap \
        >"$log" 2>&1
    echo "[$(date '+%F %T')] seed=$seed gpu=$gpu done" | tee -a "$LOG_DIR/run.log"
}

pids=()
for seed in 0 1 2 3; do
    run_seed "$seed" "$seed" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
run_seed 4 0

{
    printf 'seed\tsuccess_rate\n'
    for seed in 0 1 2 3 4; do
        rate=$(grep 'SFT overall:' "$LOG_DIR/sft_seed${seed}.log" | tail -1 | awk '{print $3}')
        printf '%s\t%s\n' "$seed" "$rate"
    done
} > "$LOG_DIR/summary.tsv"

echo "[$(date '+%F %T')] all SFT protocol evaluations done" | tee -a "$LOG_DIR/run.log"
cat "$LOG_DIR/summary.tsv"
