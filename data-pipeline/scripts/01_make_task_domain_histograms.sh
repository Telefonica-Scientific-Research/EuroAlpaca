#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
ALPACA_ROOT="${ALPACA_ROOT:-${REPO_DIR}/data/alpaca_multilingual}"

INPUT="${INPUT:-${ALPACA_ROOT}/en/alpaca_cleaned_labeled_gemma4.jsonl}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${ALPACA_ROOT}/en/analysis}"

mkdir -p "${ANALYSIS_ROOT}"

python -m alpaca_localization.make_task_domain_histograms \
  --input "${INPUT}" \
  --task-primary-out "${ANALYSIS_ROOT}/task_primary_hist.png" \
  --task-secondary-out "${ANALYSIS_ROOT}/task_secondary_hist.png" \
  --domain-primary-out "${ANALYSIS_ROOT}/domain_primary_hist.png" \
  --domain-secondary-out "${ANALYSIS_ROOT}/domain_secondary_hist.png"
