#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/validate_translation_coherence_with_gemma.log}"

GEMMA_ENDPOINT="${GEMMA_ENDPOINT:-http://127.0.0.1:10001/chat/completions}"
GEMMA_TIMEOUT="${GEMMA_TIMEOUT:-130}"
GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-8192}"
GEMMA_TEMPERATURE="${GEMMA_TEMPERATURE:-0.0}"
RESUME="${RESUME:-true}"
MAX_RECORDS="${MAX_RECORDS:-0}"
LOG_EVERY="${LOG_EVERY:-100}"
ERROR_RAW_CHARS="${ERROR_RAW_CHARS:-5000}"
QUARANTINE_FILENAME="${QUARANTINE_FILENAME:-alpaca_post_correction_quarantine.jsonl}"

DATASETS=(
  "/mnt/mirror1/aleixsant/alpaca_cleaned/de/alpaca_cleaned_labeled_annotated.jsonl"
)

mkdir -p "${LOG_ROOT}"
: > "${LOG_FILE}"

for INPUT_JSONL in "${DATASETS[@]}"; do
  if [[ ! -f "${INPUT_JSONL}" ]]; then
    echo "[warn] Missing dataset, skipping: ${INPUT_JSONL}" | tee -a "${LOG_FILE}"
    continue
  fi

  OUTPUT_JSONL="$(dirname "${INPUT_JSONL}")/alpaca_cleaned_labeled_annotated_post_validated.jsonl"
  LANG_TAG="$(sed -E 's#^.*/alpaca_cleaned/([^/]+)/.*#\1#' <<< "${INPUT_JSONL}")"

  cmd=(
    python -m alpaca_localization.validate_translation_coherence
    --input "${INPUT_JSONL}"
    --output "${OUTPUT_JSONL}"
    --language-tag "${LANG_TAG}"
    --log-every "${LOG_EVERY}"
    --max-records "${MAX_RECORDS}"
    --gemma-endpoint "${GEMMA_ENDPOINT}"
    --gemma-timeout "${GEMMA_TIMEOUT}"
    --gemma-max-tokens "${GEMMA_MAX_TOKENS}"
    --gemma-temperature "${GEMMA_TEMPERATURE}"
    --error-raw-chars "${ERROR_RAW_CHARS}"
    --quarantine-filename "${QUARANTINE_FILENAME}"
  )

  if [[ "${RESUME}" == "true" ]]; then
    cmd+=(--resume)
  fi

  "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
done
