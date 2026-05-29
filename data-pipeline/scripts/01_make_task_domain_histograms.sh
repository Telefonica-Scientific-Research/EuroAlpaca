#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

INPUT="${INPUT:-/mnt/mirror1/aleixsant/alpaca_cleaned/en/alpaca_cleaned_labeled_gemma4.jsonl}"

python -m alpaca_localization.make_task_domain_histograms \
  --input "${INPUT}"
