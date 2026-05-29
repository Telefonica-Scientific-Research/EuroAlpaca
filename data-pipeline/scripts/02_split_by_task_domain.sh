#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

INPUT="${INPUT:-/mnt/mirror1/aleixsant/alpaca_cleaned/en/alpaca_cleaned_labeled.jsonl}"
DOMAIN_OUTPUT_DIR="${DOMAIN_OUTPUT_DIR:-/mnt/mirror1/aleixsant/alpaca_cleaned/en/domains}"
TASK_OUTPUT_DIR="${TASK_OUTPUT_DIR:-/mnt/mirror1/aleixsant/alpaca_cleaned/en/tasks}"

python -m alpaca_localization.split_by_primary_domain \
  --input "${INPUT}" \
  --output-dir "${DOMAIN_OUTPUT_DIR}"

python -m alpaca_localization.split_by_primary_task \
  --input "${INPUT}" \
  --output-dir "${TASK_OUTPUT_DIR}"
