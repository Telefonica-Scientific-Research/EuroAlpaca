#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
source "${PIPELINE_DIR}/scripts/languages.sh"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ALPACA_ROOT="${ALPACA_ROOT:-${REPO_DIR}/data/alpaca_multilingual}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/assemble_prevalidation_dataset.log}"

TRANSLATED_FILENAME="${TRANSLATED_FILENAME:-alpaca_cleaned_labeled_translated.jsonl}"
LOCALIZED_DISCARDED_FILENAME="${LOCALIZED_DISCARDED_FILENAME:-alpaca_discarded_samples_localized_gemma4.jsonl}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-alpaca_cleaned_labeled_annotated_gemma4.jsonl}"
EXPECTED_SOURCE_FILENAME="${EXPECTED_SOURCE_FILENAME:-alpaca_cleaned_labeled_annotated_gemma4.jsonl}"
EXPECTED_SOURCE_JSONL="${EXPECTED_SOURCE_JSONL:-${ALPACA_ROOT}/en/${EXPECTED_SOURCE_FILENAME}}"

LANGS=("${EUROALPACA_LANGS[@]}")
LANGS_CSV="${LANGS_CSV:-}"
if [[ -n "${LANGS_CSV}" ]]; then
  IFS=',' read -r -a LANGS <<< "${LANGS_CSV}"
fi
if [[ $# -gt 0 ]]; then
  LANGS=("$@")
fi

mkdir -p "${LOG_ROOT}"
: > "${LOG_FILE}"
FAILED=0

if [[ ! -f "${EXPECTED_SOURCE_JSONL}" ]]; then
  echo "[error] Missing stage-3 expected source file: ${EXPECTED_SOURCE_JSONL}" | tee -a "${LOG_FILE}" >&2
  exit 1
fi

for LANG in "${LANGS[@]}"; do
  TRANSLATED_JSONL="${ALPACA_ROOT}/${LANG}/${TRANSLATED_FILENAME}"
  LOCALIZED_DISCARDED_JSONL="${ALPACA_ROOT}/${LANG}/${LOCALIZED_DISCARDED_FILENAME}"
  OUTPUT_JSONL="${ALPACA_ROOT}/${LANG}/${OUTPUT_FILENAME}"

  if [[ ! -f "${TRANSLATED_JSONL}" ]]; then
    echo "[error] Missing translated file: ${TRANSLATED_JSONL}" | tee -a "${LOG_FILE}" >&2
    FAILED=1
    continue
  fi
  if [[ ! -f "${LOCALIZED_DISCARDED_JSONL}" ]]; then
    echo "[error] Missing localized discarded file: ${LOCALIZED_DISCARDED_JSONL}" | tee -a "${LOG_FILE}" >&2
    FAILED=1
    continue
  fi

  python -m alpaca_localization.assemble_prevalidation_dataset \
    --translated "${TRANSLATED_JSONL}" \
    --localized-discarded "${LOCALIZED_DISCARDED_JSONL}" \
    --expected-source "${EXPECTED_SOURCE_JSONL}" \
    --output "${OUTPUT_JSONL}" \
    --language "${LANG}" \
    2>&1 | tee -a "${LOG_FILE}"
done

exit "${FAILED}"
