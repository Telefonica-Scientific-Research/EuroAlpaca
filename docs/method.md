# Method Notes

We frame the problem as task-preserving localization of instruction data.

The objective is functional equivalence across languages: each localized sample should test the same capability and remain solvable in the target language. Depending on the sample, this can require translating fields, preserving fields or regenerating target-language content instead of translating literally.

Core steps:

1. Annotate source samples with task and domain labels.
2. Decide whether each sample is suitable for translation/localization.
3. Assign field-level policies for `instruction`, `input`, and `output`.
4. Translate policy-approved fields with NLLB.
5. Localize discarded or non-directly-translatable samples with Gemma.
6. Assemble the translated and task-localized branches for each language.
7. Validate post-localization coherence and propose targeted corrections.
8. Merge validated corrections into minimal final target-language outputs.
