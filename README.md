# Multilingual Instruction Localization

This repository contains tools for multilingual instruction localization. The current code focuses on a task-preserving localization pipeline for English Alpaca-style samples, with room for training and evaluation workflows to be added alongside the data pipeline.

Direct translation can invalidate tasks such as spelling, counting, rhyme, language identification, code handling, or examples where the answer depends on English surface form. This pipeline separates task/domain annotation, sample-level translation policy annotation, field-level translation decisions, NLLB translation, discarded-sample localization, post-translation validation, and final merging.

## Repository Layout

```text
data-pipeline/
  scripts/                  # runnable pipeline stages
  src/alpaca_localization/   # Python implementation
docs/                       # notes and method documentation
```

## Pipeline Stages

```text
01_annotate_tasks_domains.sh
01_make_task_domain_histograms.sh
02_split_by_task_domain.sh
03_annotate_translation_policies.sh
04_translate_with_nllb.sh
04_translate_with_nllb_multi_gpu.sh
04_translate_with_nllb_def_batch.sh
05_localize_discarded_samples.sh
06_validate_translation_coherence.sh
07_merge_validated_outputs.sh
```

`04_translate_with_nllb.sh` is the normal single-process runner. `04_translate_with_nllb_multi_gpu.sh` splits languages across GPUs and launches several normal runners in parallel. `04_translate_with_nllb_def_batch.sh` preserves the focused batch behavior from the original `4_run_translation_with_nllb_DEF.sh`.

## Usage

Install dependencies in your active environment:

```bash
pip install -r requirements.txt
```

Run a stage from the repository root:

```bash
bash data-pipeline/scripts/03_annotate_translation_policies.sh
```

Most scripts preserve the original default paths under `/mnt/mirror1/aleixsant/alpaca_cleaned`, but the main inputs, outputs, endpoints, and runtime options can be overridden with environment variables:

```bash
INPUT_JSONL=/path/to/input.jsonl \
OUTPUT_JSONL=/path/to/output.jsonl \
GEMMA_ENDPOINT=http://127.0.0.1:10001/chat/completions \
bash data-pipeline/scripts/03_annotate_translation_policies.sh
```

## Scope

This repository currently contains the data-localization pipeline. Fine-tuning and IFEval-style evaluation can be added later under separate top-level directories.
