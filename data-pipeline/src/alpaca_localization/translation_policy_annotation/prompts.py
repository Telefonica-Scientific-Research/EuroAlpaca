"""Prompt builders for sample and field policy annotation."""

import json
from typing import Dict, List

from .constants import LANG_NAME_BY_TAG, POLICY_FIELDS

def _format_target_desc(target_langs: List[str]) -> str:
    """Format target language names for prompts."""
    target_desc = []
    for t in target_langs:
        t_l = t.lower()
        target_desc.append(f"{t_l} ({LANG_NAME_BY_TAG.get(t_l, t_l)})")
    return ", ".join(target_desc)

def _format_policy_field_values(sample: Dict) -> str:
    """Format Alpaca policy fields in canonical order."""
    return "".join(
        f"{field}: {str(sample.get(field, ''))}\n"
        for field in POLICY_FIELDS
    )

def _format_field_policy_schema_example() -> str:
    """Format the expected field-policy JSON schema in canonical field order."""
    field_blocks = []
    for index, field in enumerate(POLICY_FIELDS):
        comma = "," if index < len(POLICY_FIELDS) - 1 else ""
        field_blocks.append(
            f'  "{field}": {{\n'
            '    "all_languages": {"should_translate": true},\n'
            '    "per_language": {}\n'
            f"  }}{comma}\n"
        )
    return "{\n" + "".join(field_blocks) + "}\n"


def build_sample_policy_prompt(
    sample: Dict,
    source_lang_name: str,
    target_langs: List[str],
) -> str:
    """Build sample policy prompt."""
    field_values_text = _format_policy_field_values(sample)
    primary_task = str(sample.get("primary_task", "unknown"))
    primary_domain = str(sample.get("primary_domain", "unknown"))
    secondary_tasks = sample.get("secondary_tasks", [])
    secondary_domains = sample.get("secondary_domains", [])
    target_desc_text = _format_target_desc(target_langs)

    return (
        "You are deciding the SAMPLE-LEVEL translation policy for a dataset row.\n"
        "This first pass is a global gate: decide whether the task can survive direct localization at all.\n"
        "A later field-level pass will decide which individual fields to translate or preserve.\n"
        f"Source language: {source_lang_name}\n"
        f"Target localization languages: {target_desc_text}\n"
        f"Primary task: {primary_task}\n"
        f"Primary domain: {primary_domain}\n"
        f"Secondary tasks: {secondary_tasks}\n"
        f"Secondary domains: {secondary_domains}\n\n"
        "Field values:\n"
        f"{field_values_text}\n"
        "Rules:\n"
        "- Keep task semantics intact.\n"
        "- Decide if this sample should be translated to other languages at all.\n"
        "- Set include_sample=false when the task depends on source-language form rather than meaning.\n"
        "- Set include_sample=false for language-form tasks (spelling, grammar, tense/voice edits, typo correction).\n"
        "- Set include_sample=false for token-level/wordplay tasks (letter counts, rhymes/rhyming/rhymes with, homophones, anagrams).\n"
        "- Set include_sample=false for idioms, proverbs, puns, mnemonics, first-letter tasks, and exact letter/character constraints.\n"
        "- Set include_sample=false when the output already violates a strict word/line/count constraint.\n"
        "- For classification, discard figurative-language detection tasks if MT may alter label logic.\n"
        "- For coding, discard mislabeled language-conjugation/grammar-only tasks.\n"
        "- For creative writing, discard strict form/sound constraints (haiku/limerick/acrostic/rhyme/wordplay).\n"
        "- For evaluation, discard English-form correction/judgment tasks (grammar/spelling/punctuation/usage).\n"
        "- For information extraction, discard grammar/syntax/token-form extraction tasks.\n"
        "- For other, discard language-form editing/spelling/punctuation/phonetic/accent tasks.\n"
        "- For paraphrasing, discard grammar/tense/voice/person/word-level constrained rewrites.\n"
        "- Set include_sample=false for English-constrained generation/rewrite tasks (e.g., 'in formal/simple/modern English', English-sentence-only constraints).\n"
        "- Set include_sample=false for English alphabet/pangram constraints.\n"
        "- Set include_sample=false for language-specific tasks (English-only constraints).\n"
        "- Keep include_sample=true for normal QA, reasoning, coding, summarization, and creative generation tasks.\n"
        "- Keep include_sample=true for paraphrasing focused on clarity/conciseness/flow (not language-form mechanics).\n"
        "- Translation tasks can still be include_sample=true; field policy will protect fixed-language fields.\n"
        "- If uncertain between a language-form constraint and normal meaning translation, choose include_sample=false.\n"
        "- If include_sample=true, reason_code must be directly_translatable and evidence must be empty.\n"
        "- If include_sample=false, choose exactly one reason_code from the allowed list and provide short evidence from the sample.\n\n"
        "Allowed reason_code values:\n"
        "- directly_translatable\n"
        "- english_grammar_or_spelling\n"
        "- rhyme_or_sound_constraint\n"
        "- letter_or_character_constraint\n"
        "- word_count_or_line_count_constraint\n"
        "- idiom_or_proverb\n"
        "- wordplay_pun_anagram_palindrome\n"
        "- mnemonic_or_initial_letters\n"
        "- language_specific_task\n"
        "- output_already_violates_instruction\n"
        "- other_not_directly_translatable\n\n"
        "Return ONLY valid JSON with exactly this schema:\n"
        "{\n"
        '  "sample": {\n'
        '    "include_sample": true,\n'
        '    "reason_code": "directly_translatable",\n'
        '    "evidence": ""\n'
        "  }\n"
        "}\n"
        "No markdown. No extra keys."
    )


def build_field_policy_prompt(
    sample: Dict,
    source_lang_name: str,
    target_langs: List[str],
    sample_policy: bool,
) -> str:
    """Build field policy prompt."""
    field_values_text = _format_policy_field_values(sample)
    field_policy_schema_example = _format_field_policy_schema_example()
    primary_task = str(sample.get("primary_task", "unknown"))
    primary_domain = str(sample.get("primary_domain", "unknown"))
    secondary_tasks = sample.get("secondary_tasks", [])
    secondary_domains = sample.get("secondary_domains", [])
    target_desc_text = _format_target_desc(target_langs)
    sample_policy_text = json.dumps(sample_policy, ensure_ascii=False)

    return (
        "You are deciding whether each field should be translated for multilingual dataset localization.\n"
        "This sample already passed sample-level inclusion checks.\n"
        "Decide in one shot for ALL provided target languages using a default plus exceptions.\n"
        f"Source language: {source_lang_name}\n"
        f"Target localization languages (included subset): {target_desc_text}\n"
        f"Sample translation policy (fixed): {sample_policy_text}\n"
        f"Primary task: {primary_task}\n"
        f"Primary domain: {primary_domain}\n"
        f"Secondary tasks: {secondary_tasks}\n"
        f"Secondary domains: {secondary_domains}\n\n"
        "Field values:\n"
        f"{field_values_text}\n"
        "Rules:\n"
        "- Keep task semantics intact.\n"
        "- Set should_translate=false only when translation would break task intent.\n"
        "- If a field is empty or missing, should_translate=false.\n"
        "- Do not set false just because task is rewriting/paraphrasing.\n"
        "- For language-specific tasks, sample-level policy already filtered unsuitable samples.\n"
        "- For classification: preserve input for language-ID/code/formula detection tasks.\n"
        "- For classification with constrained labels, preserve output labels.\n"
        "- For coding: keep code and exact literal-string transformations unchanged.\n"
        "- If output is already in a fixed language required by the task, usually do not translate it.\n"
        "- Proper names/titles may need to remain unchanged depending on task context.\n"
        "- Example: if task is translate English->French and output is French answer text, do not translate that output.\n"
        "- Example: if input contains a movie title or named entity that should remain original, avoid translating it.\n"
        "- IMPORTANT: reason is only needed when decision is false.\n"
        "- If decision is true, omit reason or leave it empty.\n"
        "- Reasons must be very brief (max 8 words).\n"
        "- If unsure, set should_translate=true.\n\n"
        "Return ONLY valid JSON with exactly this schema:\n"
        f"{field_policy_schema_example}"
        "- Use language tags exactly from target list.\n"
        "- all_languages is the default for every target language.\n"
        "- per_language contains only exceptions.\n"
        "No markdown. No extra keys."
    )

__all__ = ["build_sample_policy_prompt", "build_field_policy_prompt"]
