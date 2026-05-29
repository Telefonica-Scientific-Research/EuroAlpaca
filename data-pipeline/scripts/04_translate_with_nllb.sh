#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
export PYTHONPATH="${PIPELINE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

INPUT="${INPUT:-/mnt/mirror1/aleixsant/alpaca_cleaned/en/alpaca_cleaned_labeled_annotated.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/mirror1/aleixsant/alpaca_cleaned}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-alpaca_cleaned_labeled_translated.jsonl}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
LOG_BASENAME="${LOG_BASENAME:-translate_with_nllb.log}"

MODEL_ID="${MODEL_ID:-facebook/nllb-200-3.3B}"
SOURCE_LANG="${SOURCE_LANG:-en}"
DEVICE_MAP="${DEVICE_MAP:-cuda:1}"
DTYPE="${DTYPE:-float16}"

RECORD_BATCH_SIZE="${RECORD_BATCH_SIZE:-64}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-32}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-12}"
MAX_NEW_TOKENS_RATIO="${MAX_NEW_TOKENS_RATIO:-1.25}"
MAX_NEW_TOKENS_EXTRA="${MAX_NEW_TOKENS_EXTRA:-24}"
NUM_BEAMS="${NUM_BEAMS:-1}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.12}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-6}"
STATE_EVERY="${STATE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-200}"
MAX_RECORDS="${MAX_RECORDS:-0}"
WARN_LIMIT_PER_LANG="${WARN_LIMIT_PER_LANG:-40}"

RESUME="${RESUME:-false}"
KEEP_DISCARDED="${KEEP_DISCARDED:-false}"
DROP_POLICY_FIELDS="${DROP_POLICY_FIELDS:-false}"
DISABLE_TQDM="${DISABLE_TQDM:-false}"

LANGS=(
  be bg br bs ca co cs cy da de el es et eu fi fo fr ga gd gl gv hr hu is it ka lb lt lv mk mt nl nob nno pl pt ro ru sc sk sl sq sr sv uk
)

if [[ $# -gt 0 ]]; then
  LANGS=("$@")
fi

mkdir -p "${LOG_ROOT}"
LANG_LOG_FILES=()
for lang in "${LANGS[@]}"; do
  mkdir -p "${OUTPUT_ROOT}/${lang}"
  LANG_LOG_FILES+=("${OUTPUT_ROOT}/${lang}/${LOG_BASENAME}")
done

LOG_FILE="${LOG_ROOT}/${LOG_BASENAME}"

cmd=(
  python -m alpaca_localization.translate_with_nllb
  --input "${INPUT}"
  --output-root "${OUTPUT_ROOT}"
  --output-filename "${OUTPUT_FILENAME}"
  --model-id "${MODEL_ID}"
  --source-lang "${SOURCE_LANG}"
  --target-langs "${LANGS[@]}"
  --device-map "${DEVICE_MAP}"
  --dtype "${DTYPE}"
  --record-batch-size "${RECORD_BATCH_SIZE}"
  --text-batch-size "${TEXT_BATCH_SIZE}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --min-new-tokens "${MIN_NEW_TOKENS}"
  --max-new-tokens-ratio "${MAX_NEW_TOKENS_RATIO}"
  --max-new-tokens-extra "${MAX_NEW_TOKENS_EXTRA}"
  --num-beams "${NUM_BEAMS}"
  --repetition-penalty "${REPETITION_PENALTY}"
  --no-repeat-ngram-size "${NO_REPEAT_NGRAM_SIZE}"
  --state-every "${STATE_EVERY}"
  --log-every "${LOG_EVERY}"
  --warn-limit-per-lang "${WARN_LIMIT_PER_LANG}"
)

if [[ "${MAX_RECORDS}" -gt 0 ]]; then
  cmd+=(--max-records "${MAX_RECORDS}")
fi
if [[ "${RESUME}" == "true" ]]; then
  cmd+=(--resume)
fi
if [[ "${KEEP_DISCARDED}" == "true" ]]; then
  cmd+=(--keep-discarded)
fi
if [[ "${DROP_POLICY_FIELDS}" == "true" ]]; then
  cmd+=(--drop-policy-fields)
fi
if [[ "${DISABLE_TQDM}" == "true" ]]; then
  cmd+=(--disable-tqdm)
fi

"${cmd[@]}" 2>&1 | tee "${LOG_FILE}" "${LANG_LOG_FILES[@]}"
