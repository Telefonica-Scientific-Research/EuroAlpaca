#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
source "${PIPELINE_DIR}/scripts/languages.sh"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
ALPACA_ROOT="${ALPACA_ROOT:-${REPO_DIR}/data/alpaca_multilingual}"

LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/validate_translation_coherence_with_gemma.log}"

GEMMA_ENDPOINT="${GEMMA_ENDPOINT:-http://127.0.0.1:10001/chat/completions}"
GEMMA_TIMEOUT="${GEMMA_TIMEOUT:-130}"
GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-8192}"
GEMMA_TEMPERATURE="${GEMMA_TEMPERATURE:-0.0}"
GEMMA_MAX_ATTEMPTS="${GEMMA_MAX_ATTEMPTS:-3}"
GEMMA_BACKOFF_RETRIES="${GEMMA_BACKOFF_RETRIES:-2}"
GEMMA_BACKOFF_BASE_SLEEP="${GEMMA_BACKOFF_BASE_SLEEP:-0.8}"
RESUME="${RESUME:-true}"
MAX_RECORDS="${MAX_RECORDS:-0}"
LOG_EVERY="${LOG_EVERY:-100}"
ERROR_RAW_CHARS="${ERROR_RAW_CHARS:-5000}"
QUARANTINE_FILENAME="${QUARANTINE_FILENAME:-alpaca_post_correction_gemma4_quarantine.jsonl}"

LANGS=("${EUROALPACA_LANGS[@]}")
LANGS_CSV="${LANGS_CSV:-}"
if [[ -n "${LANGS_CSV}" ]]; then
  IFS=',' read -r -a LANGS <<< "${LANGS_CSV}"
fi
if [[ $# -gt 0 ]]; then
  LANGS=("$@")
fi

DATASETS=()
for LANG in "${LANGS[@]}"; do
  DATASETS+=("${ALPACA_ROOT}/${LANG}/alpaca_cleaned_labeled_annotated_gemma4.jsonl")
done
DATASETS_CSV="${DATASETS_CSV:-}"
if [[ -n "${DATASETS_CSV}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
fi

mkdir -p "${LOG_ROOT}"
: > "${LOG_FILE}"
FAILED=0

for INPUT_JSONL in "${DATASETS[@]}"; do
  if [[ ! -f "${INPUT_JSONL}" ]]; then
    echo "[error] Missing dataset: ${INPUT_JSONL}" | tee -a "${LOG_FILE}" >&2
    FAILED=1
    continue
  fi

  OUTPUT_JSONL="$(dirname "${INPUT_JSONL}")/alpaca_cleaned_labeled_annotated_post_validated_gemma4.jsonl"
  LANG_TAG="$(basename "$(dirname "${INPUT_JSONL}")")"

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
    --gemma-max-attempts "${GEMMA_MAX_ATTEMPTS}"
    --gemma-backoff-retries "${GEMMA_BACKOFF_RETRIES}"
    --gemma-backoff-base-sleep "${GEMMA_BACKOFF_BASE_SLEEP}"
    --error-raw-chars "${ERROR_RAW_CHARS}"
    --quarantine-filename "${QUARANTINE_FILENAME}"
  )

  if [[ "${RESUME}" == "true" ]]; then
    cmd+=(--resume)
  fi

  "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
done

exit "${FAILED}"
