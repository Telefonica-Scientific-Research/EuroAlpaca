#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ALPACA_ROOT="${ALPACA_ROOT:-${REPO_DIR}/data/alpaca_multilingual}"
SOURCE_PATH="${REPO_DIR}/data/source/alpaca_data_cleaned.json"
OUTPUT_JSONL="${OUTPUT_JSONL:-${ALPACA_ROOT}/en/alpaca_cleaned.jsonl}"
LIMIT="${LIMIT:-0}"

python -m alpaca_localization.prepare_source \
  --input "${SOURCE_PATH}" \
  --output "${OUTPUT_JSONL}" \
  --limit "${LIMIT}"
