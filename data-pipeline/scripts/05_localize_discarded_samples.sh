#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ALPACA_ROOT="${ALPACA_ROOT:-/mnt/mirror1/aleixsant/alpaca_cleaned}"
INPUT_JSONL="${INPUT_JSONL:-${ALPACA_ROOT}/en/alpaca_discarted_samples_gemma4_0-22201.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ALPACA_ROOT}}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-alpaca_discarted_samples_gemma4_0-22201_localized_with_gemma4.jsonl}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/localize_discarded_samples_with_gemma.log}"

RESUME="${RESUME:-true}"
MAX_RECORDS="${MAX_RECORDS:-0}"
LOG_EVERY="${LOG_EVERY:-100}"
DISABLE_TQDM="${DISABLE_TQDM:-false}"

GEMMA_ENDPOINT="${GEMMA_ENDPOINT:-http://127.0.0.1:9999/chat/completions}"
GEMMA_TIMEOUT="${GEMMA_TIMEOUT:-130}"
GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-4096}"
GEMMA_MAX_ATTEMPTS="${GEMMA_MAX_ATTEMPTS:-3}"
GEMMA_BACKOFF_RETRIES="${GEMMA_BACKOFF_RETRIES:-2}"
GEMMA_BACKOFF_BASE_SLEEP="${GEMMA_BACKOFF_BASE_SLEEP:-0.8}"
GEMMA_TEMPERATURE="${GEMMA_TEMPERATURE:-0.0}"
ERROR_RAW_CHARS="${ERROR_RAW_CHARS:-4000}"

DEFAULT_LANGS=(de fr es it fi ca gl eu cs nl pl)
LANGS=("${DEFAULT_LANGS[@]}")
LANGS_CSV="${LANGS_CSV:-}"
if [[ -n "${LANGS_CSV}" ]]; then
  IFS=',' read -r -a LANGS <<< "${LANGS_CSV}"
fi
if [[ $# -gt 0 ]]; then
  LANGS=("$@")
fi

mkdir -p "${LOG_ROOT}"
: > "${LOG_FILE}"

cmd=(
  python -m alpaca_localization.localize_discarded_samples
  --input "${INPUT_JSONL}"
  --output-root "${OUTPUT_ROOT}"
  --output-filename "${OUTPUT_FILENAME}"
  --target-langs "${LANGS[@]}"
  --max-records "${MAX_RECORDS}"
  --log-every "${LOG_EVERY}"
  --gemma-endpoint "${GEMMA_ENDPOINT}"
  --gemma-timeout "${GEMMA_TIMEOUT}"
  --gemma-max-tokens "${GEMMA_MAX_TOKENS}"
  --gemma-max-attempts "${GEMMA_MAX_ATTEMPTS}"
  --gemma-backoff-retries "${GEMMA_BACKOFF_RETRIES}"
  --gemma-backoff-base-sleep "${GEMMA_BACKOFF_BASE_SLEEP}"
  --gemma-temperature "${GEMMA_TEMPERATURE}"
  --error-raw-chars "${ERROR_RAW_CHARS}"
)

if [[ "${RESUME}" == "true" ]]; then
  cmd+=(--resume)
fi
if [[ "${DISABLE_TQDM}" == "true" ]]; then
  cmd+=(--disable-tqdm)
fi

"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
