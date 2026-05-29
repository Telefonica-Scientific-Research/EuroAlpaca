#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT="${INPUT:-/mnt/mirror1/aleixsant/alpaca_cleaned/en/alpaca_gemma4_sample_true_input_not_translatable_nonempty.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/mirror1/aleixsant/alpaca_cleaned}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-alpaca_gemma4_sample_true_input_not_translatable_nonempty.jsonl}"
SOURCE_LANG="${SOURCE_LANG:-en}"
GPUS="${GPUS:-0 1 2 3}"
RESUME="${RESUME:-true}"
KEEP_DISCARDED="${KEEP_DISCARDED:-false}"

LANGS=(
  hr nl ro ru sv uk fi gl lv mk et lt pl is ast bg bs ca cy da el fo fur ga gd hu ka lb lij lmo ltg mt nno nob oci sc scn sk sl sq sr szl vec
)

INPUT="${INPUT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OUTPUT_FILENAME="${OUTPUT_FILENAME}" \
SOURCE_LANG="${SOURCE_LANG}" \
GPUS="${GPUS}" \
RESUME="${RESUME}" \
KEEP_DISCARDED="${KEEP_DISCARDED}" \
bash "${SCRIPT_DIR}/04_translate_with_nllb_multi_gpu.sh" "${LANGS[@]}"
