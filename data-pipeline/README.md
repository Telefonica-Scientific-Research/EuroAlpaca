# EuroAlpaca data pipeline

The root [README](../README.md) contains installation requirements, model
prerequisites and the complete clean-user workflow. This file is a concise
stage reference.

All runners can be invoked from the repository root. They add
`data-pipeline/src` to `PYTHONPATH` automatically and use
`data/alpaca_multilingual` unless `ALPACA_ROOT` is set.

| Stage | Runner | Purpose |
|---|---|---|
| 0 | `scripts/00_prepare_source.sh` | Normalize Alpaca Cleaned and assign `sample_id`. |
| 1 | `scripts/01_annotate_tasks_domains.sh` | Add task and domain labels with Gemma. |
| 2 | `scripts/01_make_task_domain_histograms.sh`, `scripts/02_split_by_task_domain.sh` | Optional analysis outputs. |
| 3 | `scripts/03_annotate_translation_policies.sh` | Add sample and field routing policies. |
| 4 | `scripts/04_translate_with_nllb.sh` | Translate policy-approved fields with NLLB. |
| 5 | `scripts/05_localize_discarded_samples.sh` | Recreate policy-excluded tasks with Gemma. |
| 6 | `scripts/06_assemble_prevalidation_dataset.sh` | Verify completeness and join the two non-overlapping branches. |
| 7 | `scripts/07_validate_translation_coherence.sh` | Validate and correct localized samples. |
| 8 | `scripts/08_merge_validated_outputs.sh` | Produce minimal final JSONL records. |

The canonical target list is shared by `scripts/languages.sh` and
`src/alpaca_localization/languages.py`. Pass positional language tags to stages
4–8 to process a subset. The following example selects German and Catalan.

```bash
bash data-pipeline/scripts/04_translate_with_nllb.sh de ca
```

The main cross-stage variables are listed below.

| Variable | Meaning |
|---|---|
| `ALPACA_ROOT` | Root for English, intermediate and target-language data. |
| `GEMMA_ENDPOINT` | Fixed-model OpenAI-compatible chat-completions endpoint. |
| `MAX_RECORDS` | Per-stage cap for model-backed development runs. |
| `RESUME` | Resume supported stages from their output/checkpoint. |
| `LOG_ROOT` | Directory for stage logs. |

Read each runner for less frequently used model, batching and filename
overrides.

The following command inspects the routing decisions produced by stage 3.

```bash
PYTHONPATH=data-pipeline/src \
python -m alpaca_localization.annotation_statistics \
  --input data/alpaca_multilingual/en/alpaca_cleaned_labeled_annotated_gemma4.jsonl
```
