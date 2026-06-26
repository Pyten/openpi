#!/bin/bash
# RECAP Phase 1A: SFT Baseline Training
# Uses pi0_libero config with local checkpoint and LIBERO dataset

export HF_LEROBOT_HOME=/mnt/vepfs/pyten/Programs/data/lerobot_cache
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=openpi-recap

PI0_BASE=/home/ma-user/work/jiaochunxuan/ckpts/pi/pi0_base/params
PI06_DIR=/mnt/vepfs/pyten/Programs/code/pi0.6
LOG_DIR=$PI06_DIR/logs
CKPT_DIR=$PI06_DIR/checkpoints

mkdir -p $LOG_DIR $CKPT_DIR

echo "[Phase 1A] Starting SFT Baseline Training"
echo "  Checkpoint: $PI0_BASE"
echo "  Dataset: $HF_LEROBOT_HOME/physical-intelligence/libero"
echo "  Log: $LOG_DIR/phase1a_sft.log"

cd $PI06_DIR

XLA_PYTHON_CLIENT_MEM_FRACTION=0.92   $PI06_DIR/.venv/bin/python -u scripts/train.py pi0_libero     --exp-name recap_sft_baseline     --overwrite     --weight-loader.params-path $PI0_BASE     --batch-size 256     --num-train-steps 30000     --save-interval 1000     --checkpoint-base-dir $CKPT_DIR     --wandb-enabled true     2>&1 | tee $LOG_DIR/phase1a_sft.log

echo "[Phase 1A] Done"
