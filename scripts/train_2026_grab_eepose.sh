#!/bin/bash
# Fine-tune robot_arm14_eepose_action on 2026_grab eepose dataset (20k steps).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="/home/ma-user/work/pyten/root_cache/openpi/train.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}"
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

mkdir -p "${OPENPI_DATA_HOME}/big_vision" "${JAX_COMPILATION_CACHE_DIR}" "${HUGGINGFACE_HUB_CACHE}" "${WANDB_DIR}"
ln -sfn "${PALIGEMMA_TOKENIZER_PATH}" "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"

PI0_BASE="/home/ma-user/work/jiaochunxuan/ckpts/pi/pi0_base/params"
DATA_DIR="/home/ma-user/work/pyten/Programs/data/robot_data/2026_grab_train_arm14_state_eepose_action"
if [[ ! -d "${PI0_BASE}" ]]; then
  echo "Missing pi0_base checkpoint: ${PI0_BASE}"
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing dataset: ${DATA_DIR}"
  exit 1
fi

cd "${ROOT_DIR}"
PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing venv python: ${PYTHON}"
  exit 1
fi

echo "W&B mode: ${WANDB_MODE} (API key set: $([[ -n \"${WANDB_API_KEY:-}\" ]] && echo yes || echo no))"

exec "${PYTHON}" scripts/train.py robot_arm14_eepose_action \
  --exp-name=2026_grab_eepose_20k \
  --overwrite \
  --num-train-steps=20000 \
  --batch-size=8 \
  --data.repo-id="${DATA_DIR}" \
  --checkpoint-base-dir="${ROOT_DIR}/checkpoints" \
  --fsdp-devices=1 \
  "$@"
