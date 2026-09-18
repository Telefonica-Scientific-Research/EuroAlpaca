#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
ALPACA_ROOT="${ALPACA_ROOT:-${REPO_DIR}/data/alpaca_multilingual}"

INPUT="${INPUT:-${ALPACA_ROOT}/en/alpaca_cleaned_labeled_gemma4.jsonl}"
DOMAIN_OUTPUT_DIR="${DOMAIN_OUTPUT_DIR:-${ALPACA_ROOT}/en/domains}"
TASK_OUTPUT_DIR="${TASK_OUTPUT_DIR:-${ALPACA_ROOT}/en/tasks}"

python -m alpaca_localization.split_by_primary_domain \
  --input "${INPUT}" \
  --output-dir "${DOMAIN_OUTPUT_DIR}"

python -m alpaca_localization.split_by_primary_task \
  --input "${INPUT}" \
  --output-dir "${TASK_OUTPUT_DIR}"
