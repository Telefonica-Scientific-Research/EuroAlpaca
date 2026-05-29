"""Sample-level deterministic guardrails."""

import re
from typing import Dict, List

from .constants import (
    CLASSIFICATION_SHORT_LABELS,
    CONSTRAINED_LABELS,
    LANGUAGE_ALIAS_TO_TAG,
    POLICY_FIELDS,
)
from .normalize import _sample_policy_metadata
from . import regex_banks as rb

from .task_detectors import (
    _is_generated_identifier_task,
    _is_list_or_array_transform_task,
    _is_search_or_catalog_alphabetical_task,
)
from .utils import _as_text, _matches_any, _reason_if_matches, _word_count

def _looks_like_code(text: str) -> bool:
    """Detect whether text looks like code."""
    t = text.strip()
    if not t:
        return False
    if "```" in t:
        return True

    # High-confidence code patterns only. Avoid broad lexical markers such as
    # "return", "class", "select", "update" that commonly appear in prose.
    strong_single_line_patterns = [
        r"^\s*#include\s+[<\"]",
        r"^\s*(?:from\s+\w+\s+import\s+\w+|import\s+\w+)",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+\s*[:\(]",
        r"\bfunction\s+\w+\s*\(",
        r"\b(?:const|let|var)\s+\w+\s*=",
        r"\bpublic\s+static\s+void\s+main\s*\(",
        r"(?i)\bSELECT\b.{0,80}\bFROM\b",
        r"(?i)\bINSERT\s+INTO\b",
        r"(?i)\bUPDATE\s+\w+\s+SET\b",
        r"(?i)\bCREATE\s+TABLE\b",
        r"(?i)\bALTER\s+TABLE\b",
        r"(?i)\bFOREIGN\s+KEY\s*\(",
        r"^\s*<html\b",
        r"<(?:/?(?:p|div|span|form|label|input|table|tr|td|th|strong|em|body|head|title|h[1-6])\b|!DOCTYPE\s+html)",
        r"^\s*struct\s+\w+\s*\{",
        r"\b(?:int|float|double|string|char|bool|boolean)\s+\w+(?:\[\])?\s*[=;]",
        r"\b\w+\s*=\s*[\"']?[^\n]+",
        r"^\s*[\[{].*[\]}]\s*$",
        r"^\s*if\s+.+:\s*$",
        r"^\s*if\s*\(.+\)\s*\{?",
        r"^\s*elif\s+.+:\s*$",
        r"^\s*else:\s*$",
        r"^\s*for\s+\w+\s+in\s+.+:\s*$",
        r"^\s*for\s*\(.+\)\s*\{?",
        r"^\s*while\s+.+:\s*$",
    ]
    for pat in strong_single_line_patterns:
        if re.search(pat, t, flags=re.MULTILINE | re.DOTALL):
            return True

    return False


def _is_constrained_label_output(text: str) -> bool:
    """Return whether the input matches constrained label output."""
    t = text.strip().lower()
    if not t:
        return False
    if t in CONSTRAINED_LABELS:
        return True
    if re.fullmatch(r"[0-9]+", t):
        return True
    return False


def _has_language_sensitive_cue(instruction: str) -> bool:
    """Return whether the input has language sensitive cue."""
    ins = instruction.lower()
    return any(cue in ins for cue in rb.LANGUAGE_SENSITIVE_CUES)


def _find_mentioned_language_tags(text: str) -> List[str]:
    """Find mentioned language tags in the input text."""
    lowered = text.lower()
    found = set()
    for alias, tag in LANGUAGE_ALIAS_TO_TAG.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            found.add(tag)
    return sorted(found)


def _classification_sample_discard_reason(instruction: str) -> str:
    """Apply classification-task guardrail logic."""
    ins = instruction.lower()
    return (
        _reason_if_matches(rb.CLASSIFICATION_SAMPLE_DISCARD_REGEXES, ins, "classification language form")
        or _reason_if_matches(rb.CLASSIFICATION_REVIEW_DISCARD_REGEXES, ins, "classification figurative task")
    )


def _classification_input_protect_reason(instruction: str) -> str:
    """Apply classification-task guardrail logic."""
    ins = instruction.lower()
    if re.search(rb.CLASSIFICATION_INPUT_PROTECT_EXCLUDE_REGEX, ins):
        return ""
    return _reason_if_matches(rb.CLASSIFICATION_INPUT_PROTECT_REGEXES, ins, "classification source signal")


def _is_classification_short_label_output(text: str) -> bool:
    """Return whether the input matches classification short label output."""
    t = " ".join(text.strip().lower().split()).strip(".").strip('"').strip("'")
    if not t:
        return False
    if t in CLASSIFICATION_SHORT_LABELS:
        return True
    if re.fullmatch(r"[0-9]+", t):
        return True
    if len(t.split()) <= 1 and len(t) <= 18 and re.fullmatch(r"[a-z0-9\-]+", t):
        return True
    return False


def _classification_output_protect_reason(instruction: str, output_text: str) -> str:
    """Apply classification-task guardrail logic."""
    ins = instruction.lower()
    if _matches_any(rb.CLASSIFICATION_OUTPUT_PROTECT_REGEXES, ins):
        return "classification fixed labels"
    return "classification fixed labels" if _is_classification_short_label_output(output_text) else ""


def _coding_sample_discard_reason(instruction: str) -> str:
    """Apply coding-task guardrail logic."""
    ins = instruction.lower()
    if re.search(rb.CODING_SAMPLE_DISCARD_EXCLUDE_REGEX, ins):
        return ""
    return _reason_if_matches(rb.CODING_SAMPLE_DISCARD_REGEXES, ins, "coding non-code language task")


def _coding_has_literal_payload(instruction: str, input_text: str) -> bool:
    """Apply coding-task guardrail logic."""
    if re.search(r'"[^"\n]{1,}"|\'[^\'\n]{1,}\'|“[^”\n]{1,}”|‘[^’\n]{1,}’', instruction):
        return True
    if isinstance(input_text, str) and input_text.strip() and any(ch.isalpha() for ch in input_text):
        return True
    return False


def _coding_is_literal_string_task(instruction: str, input_text: str) -> bool:
    """Apply coding-task guardrail logic."""
    ins = instruction.lower()
    if not _coding_has_literal_payload(instruction, input_text):
        return False
    return any(re.search(pat, ins) for pat in rb.CODING_LITERAL_TASK_REGEXES)


def _is_short_output_text(text: str) -> bool:
    """Return whether the input matches short output text."""
    t = " ".join(text.strip().split())
    if not t:
        return False
    if len(t) <= 160:
        return True
    if t.count("\n") <= 2 and len(t.split()) <= 30:
        return True
    return False


def _coding_input_protect_reason(instruction: str, input_text: str) -> str:
    """Apply coding-task guardrail logic."""
    if _coding_is_literal_string_task(instruction, input_text):
        if isinstance(input_text, str) and input_text.strip():
            return "coding literal input"
    return ""


def _coding_output_protect_reason(instruction: str, input_text: str, output_text: str) -> str:
    """Apply coding-task guardrail logic."""
    if not _coding_is_literal_string_task(instruction, input_text):
        return ""
    if _looks_like_code(output_text):
        return ""
    ins = instruction.lower()
    if any(re.search(pat, ins) for pat in rb.CODING_LITERAL_OUTPUT_DIRECTIVE_REGEXES):
        return "coding literal output"
    if _is_short_output_text(output_text):
        return "coding literal output"
    return ""


def _creative_writing_sample_discard_reason(instruction: str) -> str:
    """Apply creative-writing guardrail logic."""
    return _reason_if_matches(
        rb.CREATIVE_WRITING_SAMPLE_DISCARD_REGEXES,
        instruction.lower(),
        "creative writing form constraint",
    )


def _evaluation_sample_discard_reason(instruction: str) -> str:
    """Apply evaluation-task guardrail logic."""
    ins = instruction.lower()
    has_strong_language_form_cue = _matches_any(rb.EVALUATION_SAMPLE_DISCARD_STRONG_REGEXES, ins)
    if not has_strong_language_form_cue and re.search(rb.EVALUATION_SAMPLE_DISCARD_EXCLUDE_REGEX, ins):
        return ""
    return _reason_if_matches(rb.EVALUATION_SAMPLE_DISCARD_REGEXES, ins, "evaluation language form")


def _evaluation_output_protect_reason(instruction: str, output_text: str) -> str:
    """Apply evaluation-task guardrail logic."""
    ins = instruction.lower()
    if _matches_any(rb.EVALUATION_OUTPUT_PROTECT_REGEXES, ins):
        return "evaluation fixed labels"
    return "evaluation fixed labels" if _is_constrained_label_output(output_text) else ""


def _information_extraction_sample_discard_reason(instruction: str) -> str:
    """Apply information-extraction guardrail logic."""
    ins = instruction.lower()
    has_strong_language_form_cue = _matches_any(rb.INFORMATION_EXTRACTION_SAMPLE_DISCARD_STRONG_REGEXES, ins)
    if (
        not has_strong_language_form_cue
        and re.search(rb.INFORMATION_EXTRACTION_SAMPLE_DISCARD_EXCLUDE_REGEX, ins)
    ):
        return ""
    return _reason_if_matches(rb.INFORMATION_EXTRACTION_SAMPLE_DISCARD_REGEXES, ins, "info extraction language form")


def _other_sample_discard_reason(instruction: str) -> str:
    """Handle other sample discard reason."""
    ins = instruction.lower()
    has_strong_language_form_cue = _matches_any(rb.OTHER_SAMPLE_DISCARD_STRONG_REGEXES, ins)
    if not has_strong_language_form_cue and re.search(rb.OTHER_SAMPLE_DISCARD_EXCLUDE_REGEX, ins):
        return ""
    return _reason_if_matches(rb.OTHER_SAMPLE_DISCARD_REGEXES, ins, "other language form")


def _other_output_protect_reason(instruction: str, output_text: str) -> str:
    """Handle other output protect reason."""
    ins = instruction.lower()
    if _matches_any(rb.OTHER_OUTPUT_PROTECT_REGEXES, ins):
        return "other fixed format"
    # Preserve literal URL outputs.
    if re.search(r"https?://|www\.", output_text, flags=re.IGNORECASE):
        return "other fixed format"
    # Preserve symbolic-math outputs when the task is format-constrained.
    if re.search(r"\b\d+\s*/\s*\d+\b|\bx\s*10\^|\^|y\s*=|x\s*[<>]=?\s*\d+", output_text, flags=re.IGNORECASE):
        if re.search(r"\b(fraction|scientific notation|equation|inequality)\b", ins):
            return "other fixed format"
    return ""


def _paraphrasing_sample_discard_reason(instruction: str) -> str:
    """Apply paraphrasing-task guardrail logic."""
    ins = instruction.lower()
    has_strong_language_form_cue = _matches_any(rb.PARAPHRASING_SAMPLE_DISCARD_STRONG_REGEXES, ins)
    if not has_strong_language_form_cue and re.search(rb.PARAPHRASING_SAMPLE_DISCARD_EXCLUDE_REGEX, ins):
        return ""
    return _reason_if_matches(rb.PARAPHRASING_SAMPLE_DISCARD_REGEXES, ins, "paraphrasing language form")


def _procedural_sample_discard_reason(instruction: str, input_text: str) -> str:
    """Apply procedural-task guardrail logic."""
    text = f"{instruction}\n{input_text}".lower()
    return _reason_if_matches(rb.PROCEDURAL_SAMPLE_DISCARD_REGEXES, text, "procedural language form")


def _question_answering_sample_discard_reason(instruction: str, input_text: str) -> str:
    """Apply question-answering guardrail logic."""
    text = f"{instruction}\n{input_text}".lower()
    if re.search(rb.QUESTION_ANSWERING_SAMPLE_DISCARD_EXCLUDE_REGEX, text):
        return ""
    return _reason_if_matches(rb.QUESTION_ANSWERING_SAMPLE_DISCARD_REGEXES, text, "qa language form")


def _question_answering_output_protect_reason(instruction: str, input_text: str) -> str:
    """Apply question-answering guardrail logic."""
    text = f"{instruction}\n{input_text}".lower()
    mentioned_tags = _find_mentioned_language_tags(text)
    if not mentioned_tags:
        return ""

    if _matches_any(rb.QUESTION_ANSWERING_OUTPUT_LANGUAGE_LOCK_REGEXES, text):
        return "qa target-language answer"

    # Also protect lexical requests tied to a non-English language (e.g., adjectives in Spanish).
    if any(tag != "en" for tag in mentioned_tags):
        if re.search(
            r"\b(words?|phrases?|adjectives?|verbs?|nouns?|greetings?)\b.{0,45}\b(in|into)\b",
            text,
        ):
            return "qa target-language answer"

    return ""


def _fixed_language_output_protect_reason(instruction: str, output_text: str) -> str:
    """Handle fixed language output protect reason."""
    text = instruction.lower()
    if not _find_mentioned_language_tags(text):
        return ""
    if _matches_any(rb.QUESTION_ANSWERING_OUTPUT_LANGUAGE_LOCK_REGEXES, text):
        return "fixed-language answer"
    if re.search(r"\b(?:name|word|phrase|sentence|sentences)\b.{0,80}\b(?:in|using)\s+(?:french|spanish|german|italian|catalan|portuguese)\b", text):
        return "fixed-language answer"
    if re.search(r"\b(?:french|spanish|german|italian|catalan|portuguese)\s+(?:name|word|phrase|sentence|sentences)\b", text):
        return "fixed-language answer"
    return ""


def _reasoning_sample_discard_reason(instruction: str, input_text: str) -> str:
    """Apply reasoning-task guardrail logic."""
    text = f"{instruction}\n{input_text}".lower()
    if re.search(rb.REASONING_SAMPLE_DISCARD_EXCLUDE_REGEX, text):
        return ""
    return _reason_if_matches(rb.REASONING_SAMPLE_DISCARD_REGEXES, text, "reasoning language form")


def _style_transfer_sample_discard_reason(instruction: str, input_text: str) -> str:
    """Apply style-transfer guardrail logic."""
    text = f"{instruction}\n{input_text}".lower()
    return _reason_if_matches(rb.STYLE_TRANSFER_SAMPLE_DISCARD_REGEXES, text, "style transfer language form")


def _summarization_sample_discard_reason(instruction: str) -> str:
    """Apply summarization-task guardrail logic."""
    return _reason_if_matches(
        rb.SUMMARIZATION_SAMPLE_DISCARD_REGEXES,
        instruction.lower(),
        "summarization format constraint",
    )

def _surface_form_sample_discard_reason(instruction: str, input_text: str, output_text: str) -> str:
    """Catch high-confidence source-language form constraints across tasks."""
    text = f"{instruction}\n{input_text}".lower()

    if _matches_any(rb.GIVEN_WORD_FORM_REGEXES, text):
        return "given words form constraint"

    if _matches_any(rb.SOURCE_SENTENCE_TRANSFORM_REGEXES, text):
        return "sentence transformation constraint"

    if _matches_any(rb.LETTER_CONSTRAINT_REGEXES, text):
        return "letter constraint"

    if _matches_any(rb.LEXICAL_FORM_TASK_REGEXES, text):
        return "lexical form task"

    if _matches_any(rb.CHARACTER_LIMIT_REGEXES, text):
        return "character limit"

    if _matches_any(rb.SURFACE_FORM_DISCARD_REGEXES, text):
        return "surface form constraint"

    # Maximum word-count constraints are fragile under direct MT. If the English
    # output already violates the max, the sample is broken regardless of target.
    max_text = re.sub(r"\bno\s+less\s+than\s+\d+\s+words?\b", "", text)
    for pat in rb.MAX_WORD_COUNT_REGEXES:
        for match in re.finditer(pat, max_text):
            limit = int(match.group(1))
            if _word_count(output_text) >= limit:
                return "output violates word count"
            return "strict word count"

    for pat in rb.EXACT_WORD_COUNT_REGEXES:
        for match in re.finditer(pat, text):
            return "strict word count"

    for pat in rb.TEXT_WORD_COUNT_REGEXES:
        for match in re.finditer(pat, text):
            if rb.TEXT_NUMBER_BY_WORD.get(match.group(1), 0) > 0:
                return "strict word count"

    if _matches_any(rb.LINE_COUNT_REGEXES, text):
        return "line count constraint"

    return ""

def _looks_like_translation_intent(instruction: str) -> bool:
    """Return True when the instruction appears to request translation/localization."""
    ins = instruction.strip().lower()
    if re.search(r"\btranslat\w*\b", ins):
        return True
    if re.search(r"\blocali[sz]e\b", ins):
        return True
    if re.search(r"\bhow do you say\b", ins):
        return True
    if re.search(r"\bhow would you\b.{0,40}\bin\s+[a-z][a-z ]+\b", ins):
        return True
    if re.search(r"\b(?:spanish|french|german|italian)\s+equivalent\b", ins):
        return True
    if re.search(r"\b(?:translation|equivalent)\b", ins):
        return True
    return False


def _translation_sample_discard_reason(instruction: str, input_text: str) -> str:
    """Apply translation-task language guardrail logic."""
    text = f"{instruction}\n{input_text}".lower()

    if _matches_any(rb.TRANSLATION_NON_LANGUAGE_CONVERSION_REGEXES, text):
        return "translation non-language conversion"

    mentioned_tags = _find_mentioned_language_tags(text)
    has_explicit_language_reference = bool(mentioned_tags)
    if re.search(rb.TRANSLATION_AMBIGUOUS_TARGET_REGEX, text) and not has_explicit_language_reference:
        return "translation ambiguous target"

    if re.search(rb.TRANSLATION_ENGLISH_VARIANT_REGEX, text):
        return "translation english variant"

    # Keep lexical translation prompts ("How do you say...", "X translation"), but
    # discard only weak generation prompts without explicit language reference.
    if not _looks_like_translation_intent(instruction):
        if re.search(rb.TRANSLATION_LANGUAGE_GENERATION_REGEX, text) and not has_explicit_language_reference:
            return "translation language generation"

    return ""

def _english_specific_sample_discard_reason(
    instruction: str,
    input_text: str,
    task: str,
) -> str:
    """Detect English-specific tasks that should not be directly localized."""
    if task in {"translation", "coding"}:
        return ""

    # Use instruction-first detection to avoid false positives from long contextual inputs.
    # Example: coding inputs may mention "English alphabet" without making the task
    # English-specific in a way that breaks MT localization.
    text = instruction.lower()
    has_english_marker = _matches_any(rb.ENGLISH_SPECIFIC_MARKER_REGEXES, text)
    if not has_english_marker:
        return ""

    # Keep translation-intent prompts out of this guardrail; they are handled separately.
    if _looks_like_translation_intent(instruction):
        return ""

    if re.search(r"\b(english[- ]only|only english)\b", text):
        return "language specific task"

    if _matches_any(rb.ENGLISH_SPECIFIC_FORM_CUE_REGEXES, text):
        return "english-specific language form"

    return ""


TASK_SAMPLE_DISCARD_RULES = {
    "classification": (lambda instruction, input_text, output_text: _classification_sample_discard_reason(instruction), False),
    "creative_writing": (lambda instruction, input_text, output_text: _creative_writing_sample_discard_reason(f"{instruction}\n{input_text}"), True),
    "evaluation": (lambda instruction, input_text, output_text: _evaluation_sample_discard_reason(instruction), False),
    "information_extraction": (lambda instruction, input_text, output_text: _information_extraction_sample_discard_reason(instruction), False),
    "other": (lambda instruction, input_text, output_text: _other_sample_discard_reason(instruction), False),
    "paraphrasing": (lambda instruction, input_text, output_text: _paraphrasing_sample_discard_reason(instruction), True),
    "procedural_instructions": (lambda instruction, input_text, output_text: _procedural_sample_discard_reason(instruction, input_text), True),
    "question_answering": (lambda instruction, input_text, output_text: _question_answering_sample_discard_reason(instruction, input_text), True),
    "reasoning": (lambda instruction, input_text, output_text: _reasoning_sample_discard_reason(instruction, input_text), True),
    "style_transfer": (lambda instruction, input_text, output_text: _style_transfer_sample_discard_reason(instruction, input_text), True),
    "summarization": (lambda instruction, input_text, output_text: _summarization_sample_discard_reason(instruction), True),
    "translation": (lambda instruction, input_text, output_text: _translation_sample_discard_reason(instruction, input_text), True),
}


def _detect_mt_unsafe_sample_reason(sample: Dict) -> str:
    """Return a deterministic discard reason, or empty string when no guardrail fires."""
    field_text = {
        field: _as_text(sample.get(field, ""))
        for field in POLICY_FIELDS
    }
    instruction = field_text["instruction"]
    input_text = field_text["input"]
    output_text = field_text["output"]
    task = str(sample.get("primary_task", "")).strip().lower()
    ins = instruction.lower()

    english_specific_reason = _english_specific_sample_discard_reason(
        instruction=instruction,
        input_text=input_text,
        task=task,
    )
    if english_specific_reason:
        return english_specific_reason

    # Generated credentials/usernames/random strings are localizable as tasks;
    # field guardrails preserve the generated literal output.
    if _is_generated_identifier_task(instruction):
        return ""

    if re.search(
        r"\b(reorder|sort|arrange|rearrange|order)\b.{0,80}\blist\b.{0,80}\b(alphabetically|alphabetical\s+order)\b",
        ins,
    ):
        return "surface form constraint"

    if (
        re.search(r"\breorder\b.{0,80}\blist\b.{0,80}\bcriterion\b", ins)
        or re.search(r"\blist\b.{0,80}\breorder\b.{0,80}\bcriterion\b", ins)
    ):
        return "surface form constraint"

    if re.search(r"\blist\s+all\b.{0,80}\balphabetical(?:ly)?\s+order\b", ins):
        return "surface form constraint"

    if re.search(r"\blist\s+of\s+names\b.{0,80}\balphabetical order\b", ins):
        return "surface form constraint"

    if (
        re.search(r"\bsort\b.{0,80}\b(?:list\s+of\s+)?words?\b", ins)
        or re.search(r"\bsort\b.{0,80}\bsentence\b.{0,80}\bwords?\b", ins)
    ):
        return "surface form constraint"

    if re.search(r"\bsubject\s*:.*\bverb\s*:.*\bobject\s*:", f"{instruction}\n{input_text}", flags=re.IGNORECASE | re.DOTALL):
        return "given words form constraint"

    # Data/list/array transforms remain useful after localization when the
    # literal payload fields are preserved.
    if _is_list_or_array_transform_task(instruction, input_text):
        return ""

    if _is_search_or_catalog_alphabetical_task(instruction):
        return ""

    if re.search(r"\b\d+\s*-\s*word\s+code\b|\b\d+\s+word\s+code\b", ins):
        if re.search(r"\b(?:easy to remember|difficult to guess|hard to guess|memorable)\b", ins):
            return "strict word count"

    if task == "coding":
        if re.search(r"\bmorse code\b", ins):
            return "surface form constraint"
        if re.search(r"\bdetect misspelled words?\b", ins):
            return "surface form constraint"
        if re.search(r"\bwords?\s+with\s+exactly\s+\w+\s+letters?\b", ins):
            return "surface form constraint"
        if re.search(r"\bmake\b.{0,50}\bfirst word\b.{0,50}\buppercase\b", ins):
            return "surface form constraint"
        if re.search(r"\bstrings?\s+s\s+and\s+t\b", ins) and re.search(r"\badd one more letter\b|\bletter that was added\b", ins):
            return "surface form constraint"
        if re.search(r"\bconvert\b.{0,80}\bstring\b.{0,80}\b(?:lowercase|uppercase|camel[_ ]?case|snake[_ ]?case)\b", ins):
            return "surface form constraint"
        coding_reason = _coding_sample_discard_reason(instruction)
        if coding_reason:
            return coding_reason
        # Coding samples often contain exact literal strings or regex/code
        # constraints. Those are handled at field level by preserving input or
        # output, so avoid the broad surface-form sample gate here.
        return ""

    surface_form_reason = _surface_form_sample_discard_reason(
        instruction=instruction,
        input_text=input_text,
        output_text=output_text,
    )
    if surface_form_reason:
        return surface_form_reason

    task_rule = TASK_SAMPLE_DISCARD_RULES.get(task)
    if task_rule:
        rule, stop_after_task_rule = task_rule
        task_reason = rule(instruction, input_text, output_text)
        if task_reason or stop_after_task_rule:
            return task_reason

    # Translation prompts in other tasks are handled at field-policy level.
    if _looks_like_direct_translation_prompt(instruction):
        return ""

    if _has_language_sensitive_cue(instruction):
        return "language form task"

    if _matches_any(rb.SAMPLE_DISCARD_REGEXES, ins):
        return "language form task"

    if re.search(r"\b(english[- ]only|only english)\b", ins):
        return "language specific task"
    if re.search(r"\b(in|into)\s+english\b", ins):
        return "language specific task"
    return ""


def _looks_like_direct_translation_prompt(instruction: str) -> bool:
    """Detect whether text looks like direct translation prompt."""
    ins = instruction.strip().lower()
    if ins.startswith("translate "):
        return True
    if "translate the following" in ins:
        return True
    if "translate" in ins and " into " in ins:
        return True
    return False


def _translation_has_explicit_source_language(instruction: str) -> bool:
    """Apply translation-task language guardrail logic."""
    ins = instruction.lower()
    mentioned_tags = _find_mentioned_language_tags(ins)

    # Usually means a source->target pair is explicit.
    if len(mentioned_tags) >= 2:
        return True

    # Explicit "from <language> ... (to|into) ..."
    source_span = re.search(r"\bfrom\b(?P<src>.{0,120}?)(?:\bto\b|\binto\b|$)", ins)
    if source_span and _find_mentioned_language_tags(source_span.group("src")):
        return True

    # Explicit source-language descriptors like "English sentence/text".
    text_type_suffix = (
        r"(sentence|text|paragraph|passage|article|document|phrase|word|words|"
        r"content|statement|question)"
    )
    for alias in LANGUAGE_ALIAS_TO_TAG:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)\s+{text_type_suffix}\b", ins):
            return True

    if "source language" in ins:
        return True
    return False


def _find_translation_target_language_tags(instruction: str) -> List[str]:
    """Find translation target language tags in the input text."""
    ins = instruction.lower()
    targets = set()
    for alias, tag in LANGUAGE_ALIAS_TO_TAG.items():
        if re.search(rf"\b(?:to|into)\s+{re.escape(alias)}\b", ins):
            targets.add(tag)
        elif re.search(rf"\btranslate\b.{0,80}\bin\s+{re.escape(alias)}\b", ins):
            targets.add(tag)
        elif re.search(rf"\b{re.escape(alias)}\s+translation\b", ins):
            targets.add(tag)
        elif re.search(rf"\btranslation\b.{0,35}\b(?:for|in|into|to)\s+{re.escape(alias)}\b", ins):
            targets.add(tag)
        elif re.search(rf"\bequivalent\b.{0,35}\b(?:in|into|to)\s+{re.escape(alias)}\b", ins):
            targets.add(tag)
        elif re.search(rf"\bhow do you say\b.{0,80}\bin\s+{re.escape(alias)}\b", ins):
            targets.add(tag)
        elif re.search(rf"\bhow would you\b.{0,80}\bin\s+{re.escape(alias)}\b", ins):
            targets.add(tag)
        elif re.search(rf"\blocali[sz]e\b.{0,80}\b(?:for|to|into|in)\s+{re.escape(alias)}\b", ins):
            targets.add(tag)
        elif re.search(rf"\bin\s+{re.escape(alias)}\b", ins) and _looks_like_translation_intent(instruction):
            targets.add(tag)

    # Fallback: direct translation prompt with exactly one mentioned language.
    if not targets and _looks_like_translation_intent(instruction):
        mentioned_tags = _find_mentioned_language_tags(ins)
        if len(mentioned_tags) == 1:
            targets.add(mentioned_tags[0])

    return sorted(targets)


def _sample_guardrail_reason_code(unsafe_reason: str, evidence: str = "") -> str:
    """Build sample-guardrail metadata."""
    reason = unsafe_reason.lower()
    evidence_l = evidence.lower()
    combined = f"{reason} {evidence_l}"
    if any(term in combined for term in ["word count", "strict word count", "character limit"]):
        return "word_count_or_line_count_constraint"
    if any(term in combined for term in ["british and american english", "american and british english", "british vs", "american vs"]):
        return "language_specific_task"
    if any(term in combined for term in ["idiom", "cat's out of the bag", "cat’s out of the bag", "proverb"]):
        return "idiom_or_proverb"
    if any(term in combined for term in ["onomatopoeia", "sound word"]):
        return "rhyme_or_sound_constraint"
    if any(term in combined for term in ["semicolon", "semi colon", "colon for emphasis", "modal verb"]):
        return "english_grammar_or_spelling"
    if "given words" in reason or any(
        term in combined
        for term in [
            "following words",
            "given words",
            "provided words",
            "these words",
            "set of words",
            "same order",
            "given order",
            "correct order",
            "word order",
        ]
    ):
        return "letter_or_character_constraint"
    if "letter constraint" in reason:
        return "letter_or_character_constraint"
    if "lexical form" in reason:
        return "english_grammar_or_spelling"
    if any(
        term in combined
        for term in [
            "grammar",
            "spelling",
            "punctuation",
            "typo",
            "misspell",
            "tense",
            "voice",
            "part of speech",
            "verb form",
            "pronoun",
            "compound sentence",
            "sentence",
        ]
    ):
        return "english_grammar_or_spelling"
    if any(term in combined for term in ["rhyme", "sound", "phonetic", "pronunciation", "homophone", "syllable"]):
        return "rhyme_or_sound_constraint"
    if "haiku" in combined:
        return "word_count_or_line_count_constraint"
    if any(term in combined for term in ["letter", "character"]):
        return "letter_or_character_constraint"
    if "line count" in combined or "format constraint" in combined:
        return "word_count_or_line_count_constraint"
    if "idiom" in combined or "proverb" in combined:
        return "idiom_or_proverb"
    if any(term in combined for term in ["wordplay", "pun", "anagram", "palindrome"]):
        return "wordplay_pun_anagram_palindrome"
    if "riddle" in combined:
        return "wordplay_pun_anagram_palindrome"
    if "mnemonic" in combined or "initial" in combined:
        return "mnemonic_or_initial_letters"
    if "language specific" in reason or "english-specific" in reason:
        return "language_specific_task"
    if "output violates" in reason:
        return "output_already_violates_instruction"
    if "surface form" in reason:
        return "letter_or_character_constraint"
    if "sentence transformation" in reason:
        return "english_grammar_or_spelling"
    if "language form" in reason:
        return "english_grammar_or_spelling"
    return "other_not_directly_translatable"


def _sample_guardrail_evidence(sample: Dict, unsafe_reason: str) -> str:
    """Build sample-guardrail metadata."""
    field_text = {
        field: str(sample.get(field, "")).strip()
        for field in POLICY_FIELDS
    }
    instruction = field_text["instruction"]
    input_text = field_text["input"]
    output_text = field_text["output"]
    haystacks = [field_text[field] for field in POLICY_FIELDS]
    patterns = [
        r"\b(?:starts?|begins?|ends?)\s+with\s+(?:the\s+)?(?:letter\s+)?[\"']?[A-Za-z][\"']?",
        r"\b(?:using|with|about)\s+(?:the\s+)?letter\s+[\"']?[A-Za-z][\"']?",
        r"\b(?:spelling|grammar|grammatical|grammatically|punctuation|typo|misspell\w*|tense|voice|part of speech|verb form)\b",
        r"\b(?:rhyme|rhyming|alliteration|acrostic|anagram|palindrome|homophone|syllables?|meter|metre)\b",
        r"\b(?:word count|line count|\d+\s+words?|\d+[- ]word|one-word|two-word|three-word|four-word|five-word|six-word|seven-word|eight-word|nine-word|ten-word)\b",
        r"\b(?:given|following|provided)?\s*(?:set of )?words?\b.{0,80}\b(?:sentence|phrase|question|interrogative)\b",
        r"\b(?:sentence|phrase|question|interrogative)\b.{0,80}\b(?:given|following|provided)?\s*(?:set of )?words?\b",
        r"\b(?:same|given|correct)\s+order\b",
        r"\b(?:reported speech|direct speech|indirect speech)\b",
        r"\b(?:contain|contains|containing)\b.{0,40}\b(?:the\s+)?letter\s+[\"']?[A-Za-z][\"']?\b",
        r"\b(?:irregular verb|root word|english vocabulary word|english synonyms?|proper noun|each word|modal verb|passive voice)\b",
        r"\b(?:idiom|onomatopoeia|british and american english|british vs\.? american english|american and british english|semicolon|semi colon|colon for emphasis)\b",
        r"\b(?:less than|under|no more than|maximum(?:\s+of)?)\s+\d+\s+characters?\b",
        r"\b(?:idiom|proverb|pun|wordplay|mnemonic)\b",
        r"\b(?:english[- ]only|only english|in english|english sentence|english word)\b",
    ]
    for text in haystacks:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)[:160]
    if instruction:
        return instruction[:160]
    if input_text:
        return input_text[:160]
    if output_text:
        return output_text[:160]
    return unsafe_reason


def apply_sample_guardrails(
    sample: Dict,
    sample_policy: Dict[str, object],
) -> Dict[str, object]:
    """Apply sample guardrails."""
    # Start from Gemma's global sample decision, then let deterministic evidence
    # veto samples that are clearly unsafe for direct MT.
    unsafe_reason = _detect_mt_unsafe_sample_reason(sample)
    if unsafe_reason:
        evidence = _sample_guardrail_evidence(sample, unsafe_reason)
        return _sample_policy_metadata(
            include_sample=False,
            reason_code=_sample_guardrail_reason_code(unsafe_reason, evidence),
            evidence=evidence,
            source="guardrail",
        )
    return sample_policy

__all__ = [
    "_looks_like_code",
    "_is_short_output_text",
    "_detect_mt_unsafe_sample_reason",
    "_surface_form_sample_discard_reason",
    "_sample_guardrail_reason_code",
    "_sample_guardrail_evidence",
    "apply_sample_guardrails",
]
