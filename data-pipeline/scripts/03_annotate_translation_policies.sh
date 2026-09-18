#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ALPACA_ROOT="${ALPACA_ROOT:-${REPO_DIR}/data/alpaca_multilingual}"
INPUT_JSONL="${INPUT_JSONL:-${ALPACA_ROOT}/en/alpaca_cleaned_labeled_gemma4.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${ALPACA_ROOT}/en/alpaca_cleaned_labeled_annotated_gemma4.jsonl}"
PARSE_FAILURE_DIR="${PARSE_FAILURE_DIR:-${ALPACA_ROOT}/en/annotation_parse_failures}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/annotate_translation_policies_with_gemma.log}"

SOURCE_LANG="${SOURCE_LANG:-en}"
RESUME="${RESUME:-true}"
LOG_EVERY="${LOG_EVERY:-100}"
MAX_RECORDS="${MAX_RECORDS:-0}"
OUTPUT_SAMPLE_POLICY_FIELD="${OUTPUT_SAMPLE_POLICY_FIELD:-sample_translation_policy}"
OUTPUT_POLICY_FIELD="${OUTPUT_POLICY_FIELD:-field_translation_policy}"
REUSE_EXISTING_POLICIES_WITHOUT_QUERYING_GEMMA="${REUSE_EXISTING_POLICIES_WITHOUT_QUERYING_GEMMA:-false}"

GEMMA_ENDPOINT="${GEMMA_ENDPOINT:-http://127.0.0.1:10001/chat/completions}"
GEMMA_TIMEOUT="${GEMMA_TIMEOUT:-130}"
GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-1000}"
GEMMA_TEMPERATURE="${GEMMA_TEMPERATURE:-0.0}"

mkdir -p "${PARSE_FAILURE_DIR}" "${LOG_ROOT}"
: > "${LOG_FILE}"

cmd=(
  python -m alpaca_localization.annotate_translation_policies
  --input "${INPUT_JSONL}"
  --output "${OUTPUT_JSONL}"
  --parse-failure-dir "${PARSE_FAILURE_DIR}"
  --output-policy-field "${OUTPUT_POLICY_FIELD}"
  --output-sample-policy-field "${OUTPUT_SAMPLE_POLICY_FIELD}"
  --source-lang "${SOURCE_LANG}"
  --log-every "${LOG_EVERY}"
  --max-records "${MAX_RECORDS}"
  --gemma-endpoint "${GEMMA_ENDPOINT}"
  --gemma-timeout "${GEMMA_TIMEOUT}"
  --gemma-max-tokens "${GEMMA_MAX_TOKENS}"
  --gemma-temperature "${GEMMA_TEMPERATURE}"
)

if [[ "${RESUME}" == "true" ]]; then
  cmd+=(--resume)
fi
if [[ "${REUSE_EXISTING_POLICIES_WITHOUT_QUERYING_GEMMA}" == "true" ]]; then
  cmd+=(--reuse-existing-policies-without-querying-gemma)
fi

"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
