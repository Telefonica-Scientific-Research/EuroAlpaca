#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
ALPACA_ROOT="${ALPACA_ROOT:-${REPO_DIR}/data/alpaca_multilingual}"

INPUT="${INPUT:-${ALPACA_ROOT}/en/alpaca_cleaned.jsonl}"
OUTPUT="${OUTPUT:-${ALPACA_ROOT}/en/alpaca_cleaned_labeled_gemma4.jsonl}"
GEMMA_ENDPOINT="${GEMMA_ENDPOINT:-http://127.0.0.1:10001/chat/completions}"
RESUME="${RESUME:-true}"

cmd=(
  python -m alpaca_localization.annotate_tasks_domains
  --input "${INPUT}"
  --output "${OUTPUT}"
  --endpoint "${GEMMA_ENDPOINT}"
)

if [[ "${RESUME}" == "true" ]]; then
  cmd+=(--resume)
fi

"${cmd[@]}"
