#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/04_translate_with_nllb.sh"

GPU_LIST_STR="${GPUS:-0 1}"
read -r -a GPUS_ARR <<< "${GPU_LIST_STR}"
if [[ ${#GPUS_ARR[@]} -eq 0 ]]; then
  echo "[error] GPUS list is empty. Example: GPUS='0 1 2 3'" >&2
  exit 1
fi

DEFAULT_LANGS=(
  be bg br bs ca co cs cy da de el es et eu fi fo fr ga gd gl gv hr hu is it ka lb lt lv mk mt nl nob nno pl pt ro ru sc sk sl sq sr sv uk
)

LANGS=("${DEFAULT_LANGS[@]}")
if [[ $# -gt 0 ]]; then
  LANGS=("$@")
fi

N_GPU=${#GPUS_ARR[@]}
declare -a ASSIGNED
for ((i=0; i<N_GPU; i++)); do
  ASSIGNED[i]=""
done

for idx in "${!LANGS[@]}"; do
  gpu_slot=$(( idx % N_GPU ))
  ASSIGNED[gpu_slot]="${ASSIGNED[gpu_slot]} ${LANGS[idx]}"
done

declare -a PIDS
FAIL=0

for ((slot=0; slot<N_GPU; slot++)); do
  gpu="${GPUS_ARR[slot]}"
  langs_str="$(echo "${ASSIGNED[slot]}" | xargs)"
  if [[ -z "${langs_str}" ]]; then
    continue
  fi

  read -r -a langs_for_gpu <<< "${langs_str}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DEVICE_MAP="cuda:0" \
    LOG_BASENAME="translate_with_nllb.gpu${gpu}.log" \
    bash "${BASE_RUNNER}" "${langs_for_gpu[@]}"
  ) &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ ${FAIL} -ne 0 ]]; then
  echo "[error] One or more GPU workers failed." >&2
  exit 1
fi
