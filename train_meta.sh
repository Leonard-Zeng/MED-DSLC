#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-training/train.py}"

DATA_ROOT="${DATA_ROOT:-./data}"
MODEL_ROOT="${MODEL_ROOT:-./output-rank2/base_split/vitb16}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-final_weights}"
META_TYPE="${META_TYPE:-MED}"
META_CONFIG_PATH="${META_CONFIG_PATH:-}"
SHOTS="${SHOTS:-16}"
MODEL_SHOTS="${MODEL_SHOTS:-16}"
SEED="${SEED:-1}"
BACKBONE="${BACKBONE:-ViT-B/16}"
DEVICE="${DEVICE:-auto}"
if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE="$("${PY}" -c 'from device_utils import get_default_device; print(get_default_device())')"
fi
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"

logit_scalar="standard"
case "${META_TYPE}" in
  MED_LCDS)
    logit_scalar="domain-wise"
    ;;
  single_lora|single_lora_from_pretrained|MED|mole|phatgoose)
    logit_scalar="standard"
    ;;
  *)
    echo "Unknown or inference-only meta training method: ${META_TYPE}" >&2
    exit 1
    ;;
esac

META_OUTPUT_DIR="${META_OUTPUT_DIR:-${WEIGHTS_ROOT}/${META_TYPE}_${logit_scalar}}"

COMMON_FLAGS=(
  --data_root "${DATA_ROOT}"
  --model_dir "${MODEL_ROOT}"
  --shots "${SHOTS}"
  --model_shots "${MODEL_SHOTS}"
  --seed "${SEED}"
  --backbone "${BACKBONE}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --weights_dir "${WEIGHTS_ROOT}"
  --output_weights_dir "${META_OUTPUT_DIR}"
)

if [[ -n "${META_CONFIG_PATH}" ]]; then
  COMMON_FLAGS+=(--meta_config_path "${META_CONFIG_PATH}")
fi

echo "[META TRAIN] method=${META_TYPE} logit_scalar=${logit_scalar} output=${META_OUTPUT_DIR}"
"${PY}" "${TRAIN_SCRIPT}" \
  --method "${META_TYPE}" \
  --logit_scalar "${logit_scalar}" \
  "${COMMON_FLAGS[@]}"

echo "Meta training completed under ${META_OUTPUT_DIR}."
