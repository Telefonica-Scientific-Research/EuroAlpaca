#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

INPUT="${INPUT:-/mnt/mirror1/aleixsant/alpaca_cleaned/en/alpaca_cleaned_0-34342.jsonl}"
OUTPUT="${OUTPUT:-/mnt/mirror1/aleixsant/alpaca_cleaned/en/alpaca_cleaned_labeled_gemma4_0-34342.jsonl}"
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
