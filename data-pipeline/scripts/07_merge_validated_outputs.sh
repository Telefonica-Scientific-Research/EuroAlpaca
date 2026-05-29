#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ALPACA_ROOT="${ALPACA_ROOT:-/mnt/mirror1/aleixsant/alpaca_cleaned}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/merge_validated_outputs.log}"
LOG_EVERY="${LOG_EVERY:-10000}"

LANGS=(es)

mkdir -p "${LOG_ROOT}"
: > "${LOG_FILE}"

for LANG in "${LANGS[@]}"; do
  ORIGINAL_JSONL="${ALPACA_ROOT}/${LANG}/alpaca_cleaned_labeled_annotated_gemma4.jsonl"
  VALIDATED_JSONL="${ALPACA_ROOT}/${LANG}/alpaca_cleaned_labeled_annotated_post_validated_gemma4.jsonl"
  OUTPUT_JSONL="${ALPACA_ROOT}/${LANG}/alpaca_cleaned_labeled_annotated_post_validated_gemma4_final.jsonl"

  if [[ ! -f "${ORIGINAL_JSONL}" ]]; then
    echo "[warn] Missing original dataset, skipping: ${ORIGINAL_JSONL}" | tee -a "${LOG_FILE}"
    continue
  fi
  if [[ ! -f "${VALIDATED_JSONL}" ]]; then
    echo "[warn] Missing validated dataset, skipping: ${VALIDATED_JSONL}" | tee -a "${LOG_FILE}"
    continue
  fi

  python -m alpaca_localization.merge_validated_outputs \
    --original "${ORIGINAL_JSONL}" \
    --validated "${VALIDATED_JSONL}" \
    --output "${OUTPUT_JSONL}" \
    --language "${LANG}" \
    --log-every "${LOG_EVERY}" \
    2>&1 | tee -a "${LOG_FILE}"
done
