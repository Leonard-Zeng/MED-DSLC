#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"

DATA_ROOT="${DATA_ROOT:-./data}"
SAVE_ROOT="${SAVE_ROOT:-./output-rank2/base_split}"
SHOTS="${SHOTS:-16}"
SEED="${SEED:-1}"
BACKBONE="${BACKBONE:-ViT-B/16}"
SUBSAMPLE="${SUBSAMPLE:-base}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-2e-4}"
R="${R:-2}"
ALPHA="${ALPHA:-1}"
DROPOUT_RATE="${DROPOUT_RATE:-0.25}"
CLASS_FILTER_PATH="${CLASS_FILTER_PATH:-}"

DOMAINS=(
  caltech101
  eurosat
  stanford_cars
  food101
  oxford_pets
  oxford_flowers
  dtd
  ucf101
  fgvc
)

COMMON_FLAGS=(
  --root_path "${DATA_ROOT}"
  --save_path "${SAVE_ROOT}"
  --shots "${SHOTS}"
  --seed "${SEED}"
  --backbone "${BACKBONE}"
  --subsample "${SUBSAMPLE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --r "${R}"
  --alpha "${ALPHA}"
  --dropout_rate "${DROPOUT_RATE}"
  --encoder both
  --position all
  --params q k v
)

if [[ -n "${CLASS_FILTER_PATH}" ]]; then
  COMMON_FLAGS+=(--class_filter_path "${CLASS_FILTER_PATH}")
fi

for domain in "${DOMAINS[@]}"; do
  echo "[TRAIN EXPERT LORA] domain=${domain}"
  "${PY}" main.py \
    --dataset "${domain}" \
    "${COMMON_FLAGS[@]}"
done

echo "Expert LoRAs saved under ${SAVE_ROOT}/vitb16."
