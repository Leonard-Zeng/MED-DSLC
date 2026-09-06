#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluation/evaluate.py}"

DATA_ROOT="${DATA_ROOT:-./data}"
MODEL_ROOT="${MODEL_ROOT:-./output-rank2/base_split/vitb16}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-trained_weights}"
OUT_ROOT="${OUT_ROOT:-results}"
META_TYPE="${META_TYPE:-MED}"
META_CONFIG_PATH="${META_CONFIG_PATH:-}"
META_WEIGHT_PATH="${META_WEIGHT_PATH:-}"
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
MAIN_FIRSTK="${MAIN_FIRSTK:-5}"

SINGLE_LORA_META_WEIGHT_PATH="${SINGLE_LORA_META_WEIGHT_PATH:-${WEIGHTS_ROOT}/single_lora_standard/single_lora.pt}"
SINGLE_LORA_PRETRAINED_META_WEIGHT_PATH="${SINGLE_LORA_PRETRAINED_META_WEIGHT_PATH:-${WEIGHTS_ROOT}/single_lora_from_pretrained_standard/single_lora.pt}"
MED_META_WEIGHT_PATH="${MED_META_WEIGHT_PATH:-${WEIGHTS_ROOT}/MED_standard}"
MED_LCDS_META_WEIGHT_PATH="${MED_LCDS_META_WEIGHT_PATH:-${WEIGHTS_ROOT}/MED_LCDS_domain-wise}"
MIXTURE_META_WEIGHT_PATH="${MIXTURE_META_WEIGHT_PATH:-${WEIGHTS_ROOT}/mole_standard}"
PHATGOOSE_META_WEIGHT_PATH="${PHATGOOSE_META_WEIGHT_PATH:-${WEIGHTS_ROOT}/phatgoose_standard}"

read -r -a BENCHMARK_MODES <<< "${BENCHMARK_MODES:-cross_domain in_domain}"
read -r -a SUBSAMPLES <<< "${SUBSAMPLES:-all base new}"

resolve_default_meta_weight_path() {
  if [[ -n "${META_WEIGHT_PATH}" ]]; then
    printf '%s\n' "${META_WEIGHT_PATH}"
    return 0
  fi

  case "${META_TYPE}" in
    single_lora) printf '%s\n' "${SINGLE_LORA_META_WEIGHT_PATH}" ;;
    single_lora_from_pretrained) printf '%s\n' "${SINGLE_LORA_PRETRAINED_META_WEIGHT_PATH}" ;;
    phatgoose) printf '%s\n' "${PHATGOOSE_META_WEIGHT_PATH}" ;;
    MED) printf '%s\n' "${MED_META_WEIGHT_PATH}" ;;
    MED_LCDS) printf '%s\n' "${MED_LCDS_META_WEIGHT_PATH}" ;;
    mole) printf '%s\n' "${MIXTURE_META_WEIGHT_PATH}" ;;
    *) printf '\n' ;;
  esac
}

meta_type_arg="${META_TYPE}"
output_name="${META_TYPE}"
extra_args=()
case "${META_TYPE}" in
  base_clip)
    output_name="base-clip"
    ;;
  expert_lora)
    output_name="expert-lora"
    ;;
  lora_mean)
    output_name="mean-lora"
    ;;
  knots_svd)
    meta_type_arg="knots_svd"
    output_name="knots-tries"
    ;;
  knots_svd_dare)
    meta_type_arg="knots_svd"
    output_name="knots-dare-tries"
    extra_args+=(--knots_dare)
    ;;
  single_lora|single_lora_from_pretrained|phatgoose|MED|MED_LCDS|mole)
    ;;
  *)
    echo "Unknown meta inference method: ${META_TYPE}" >&2
    exit 1
    ;;
esac

COMMON_FLAGS=(
  --data_root "${DATA_ROOT}"
  --model_root "${MODEL_ROOT}"
  --shots "${SHOTS}"
  --model_shots "${MODEL_SHOTS}"
  --seed "${SEED}"
  --backbone "${BACKBONE}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --enable_semantic_equivalence
  --dump_wrong_images
)

if [[ -n "${META_CONFIG_PATH}" ]]; then
  COMMON_FLAGS+=(--meta_config_path "${META_CONFIG_PATH}")
fi

meta_weight_path="$(resolve_default_meta_weight_path)"

run_eval() {
  local benchmark_mode="$1"
  local subsample="$2"
  local cmd=(
    "${PY}" "${EVAL_SCRIPT}"
    --meta_type "${meta_type_arg}"
    --benchmark_mode "${benchmark_mode}"
    --subsample "${subsample}"
    --firstk_cnt "${MAIN_FIRSTK}"
    --output_dir "${OUT_ROOT}/${output_name}/${benchmark_mode}/${subsample}"
    "${COMMON_FLAGS[@]}"
  )
  if [[ -n "${meta_weight_path}" ]]; then
    cmd+=(--meta_weight_path "${meta_weight_path}")
  fi
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi

  echo "[META INFER] method=${META_TYPE} meta_type=${meta_type_arg} mode=${benchmark_mode} subsample=${subsample}"
  "${cmd[@]}"
}

for mode in "${BENCHMARK_MODES[@]}"; do
  for sub in "${SUBSAMPLES[@]}"; do
    run_eval "${mode}" "${sub}"
  done
done

echo "Meta inference completed under ${OUT_ROOT}/${output_name}."
