"""Field-level deterministic guardrails."""

import re
from typing import Dict, List, Tuple

from .constants import POLICY_FIELDS
from .policy_mutation import FieldPolicyMutator
from .task_detectors import (
    _is_generated_identifier_task,
    _is_list_or_array_transform_task,
    _is_search_or_catalog_alphabetical_task,
)
from .sample_guardrails import (
    _classification_input_protect_reason,
    _classification_output_protect_reason,
    _coding_input_protect_reason,
    _coding_output_protect_reason,
    _evaluation_output_protect_reason,
    _fixed_language_output_protect_reason,
    _find_translation_target_language_tags,
    _is_constrained_label_output,
    _looks_like_code,
    _looks_like_direct_translation_prompt,
    _other_output_protect_reason,
    _question_answering_output_protect_reason,
    _translation_has_explicit_source_language,
)
from .utils import _as_text



def _identifier_input_should_be_preserved(input_text: str) -> bool:
    """Handle identifier input should be preserved."""
    text = input_text.strip()
    if not text:
        return False
    return bool(
        re.search(r"\b(name|user'?s name|age|combination|letters?|numbers?)\s*:", text, flags=re.IGNORECASE)
        or re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    )


def _is_regex_task(instruction: str) -> bool:
    """Return whether the input matches regex task."""
    return bool(re.search(r"\b(regex|regular expression)\b", instruction, flags=re.IGNORECASE))


def _is_cipher_or_literal_transform_task(instruction: str) -> bool:
    """Return whether the input matches cipher or literal transform task."""
    return bool(
        re.search(
            r"\b(rot13|caesar cipher|vigenere cipher|substitution cipher|cipher|cryptanalys|decipher|encode|encrypted|decrypt|encryption|binary code|ascii characters?)\b",
            instruction,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(remove all html|html elements|parse .*json|json data|reformat .*html|apply .*cipher|reverse engineer|delete all vowels|remove all vowels)\b",
            instruction,
            flags=re.IGNORECASE,
        )
    )


def _is_code_edit_task(instruction: str) -> bool:
    """Return whether the input matches code edit task."""
    return bool(
        re.search(
            r"\b(code|html|css|json|array|list|matrix|matrices|strings?|function|regex|regular expression|program|script|query|queries|sql|database|table|dictionary|algorithm)\b",
            instruction,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\b(generate|write|find|fix|edit|modify|update|optimi[sz]e|parse|reformat|rewrite|transform|remove|classify|recommend|create|return|construct|implement|design|convert|merge|sort|arrange|reverse|reflect|calculate|compute|predict|print|insert|filter|store|show)\b",
            instruction,
            flags=re.IGNORECASE,
        )
    )


def _is_literal_transform_task(instruction: str) -> bool:
    """Return whether the input matches literal transform task."""
    ins = instruction.lower()
    return bool(
        re.search(
            r"\b(reverse|normalize|delete all vowels|remove all vowels|convert|transform|reflect|merge|sort|arrange|rearrange|calculate|compute|sum|standard deviation|longest common subsequence|sub[- ]?sequence|longer one|non-duplicate|occurrences)\b",
            ins,
        )
        and re.search(
            r"\b(strings?|lists?|arrays?|matri(?:x|ces)|numbers?|integers?|binary|ascii|table|json)\b",
            ins,
        )
    )




def _is_table_json_task(instruction: str) -> bool:
    """Return whether the input matches table json task."""
    ins = instruction.lower()
    return bool(
        re.search(r"\b(?:table|csv|data)\b", ins)
        and re.search(r"\b(?:json|dictionary|dict|object)\b", ins)
    )


def _looks_like_markup_literal(text: str) -> bool:
    """Detect whether text looks like markup literal."""
    return bool(re.search(r"</?[a-z][a-z0-9-]*(?:\s[^>]*)?>", text or "", flags=re.IGNORECASE))


def _looks_like_placeholder_literal(text: str) -> bool:
    """Detect whether text looks like placeholder literal."""
    return bool(
        re.search(r"<[^>\n]{1,40}>|\[[A-Za-z_ -]{1,40}\]|_{4,}", text or "")
    )


def _looks_like_hashtag_literal(text: str) -> bool:
    """Detect whether text looks like hashtag literal."""
    return bool(re.search(r"#[A-Za-z][A-Za-z0-9_]*", text or ""))


def _looks_like_music_chord_sequence(text: str) -> bool:
    """Detect whether text looks like music chord sequence."""
    return bool(
        re.search(
            r"\b[A-G](?:#|b)?\s+(?:major|minor|maj|min)\b(?:\s*,\s*\b[A-G](?:#|b)?\s+(?:major|minor|maj|min)\b)+",
            text or "",
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b[A-G](?:#|b)?\d\b(?:\s+|\s*,\s*)\b[A-G](?:#|b)?\d\b",
            text or "",
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b[A-G]\b(?:\s*,\s*\b[A-G]\b)+(?:\s*,\s*B-flat|\s*,\s*Bb)?",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def _looks_like_foreign_phrase_payload(text: str) -> bool:
    """Detect whether text looks like foreign phrase payload."""
    return bool(re.search(r"\bbon\s*,\s*voyage\b", text or "", flags=re.IGNORECASE))


def _looks_like_boolean_expression_literal(text: str) -> bool:
    """Detect whether text looks like boolean expression literal."""
    return bool(
        re.search(r"\b[A-Z]\b\s*(?:&&|\|\||AND|OR|XOR)\s*\b[A-Z]\b", text or "", flags=re.IGNORECASE)
        or re.search(r"\b(?:true|false)\b", text or "", flags=re.IGNORECASE)
    )


def _looks_like_binary_literal(text: str) -> bool:
    """Detect whether text looks like binary literal."""
    return bool(re.search(r"\b0b[01]+\b", text or ""))


def _looks_like_command_invocation(text: str) -> bool:
    """Detect whether text looks like command invocation."""
    return bool(
        re.search(r"^\s*(?:g\+\+|gcc|clang|python3?|node|npm|yarn|pip|java|javac|cargo|go|rustc)(?=\s|$)", text or "", flags=re.MULTILINE)
        or re.search(r"`\s*(?:cd|ls|mkdir|cp|mv|rm|pwd)\s+[^`]+`", text or "")
    )


def _looks_like_filesystem_path_literal(text: str) -> bool:
    """Detect whether text looks like filesystem path literal."""
    return bool(
        re.search(r"(?:^|[\s`'\"])(?:~|/)[A-Za-z0-9_./ -]+", text or "")
        or re.search(r"\b[A-Za-z]:\\[^\s`'\"]+", text or "")
    )


def _looks_like_compact_math_literal(text: str) -> bool:
    """Detect whether text looks like compact math literal."""
    return bool(
        re.search(r"\b\d+!\s*[+\-*/]\s*\d+!|\b\d+\s*=\s*[^.\n]+|[A-Za-z]\s*=\s*[^.\n]+", text or "")
        or re.search(r"\d+\s*[*^]\s*\d+|\d+\s*/\s*\d+", text or "")
    )


def _looks_like_player_label_literal(text: str) -> bool:
    """Detect whether text looks like player label literal."""
    return bool(re.search(r"\bPlayer\s*#\d+\b", text or "", flags=re.IGNORECASE))


def _looks_like_hex_color_literal(text: str) -> bool:
    """Detect whether text looks like hex color literal."""
    return bool(re.search(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b", text or ""))


def _looks_like_dna_sequence_literal(text: str) -> bool:
    """Detect whether text looks like dna sequence literal."""
    return bool(re.search(r"\b[ACGT]{20,}\b", text or ""))


def _looks_like_objective_label_literal(text: str) -> bool:
    """Detect whether text looks like objective label literal."""
    return bool(
        re.search(r"@\w+", text or "")
        or re.search(r"\bObjective\s*:\s*(?:Promotion|Complaint|Information|Other)\b", text or "")
    )


def _looks_like_url_or_contact_literal(text: str) -> bool:
    """Detect whether text looks like url or contact literal."""
    return bool(
        re.search(r"https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,}", text or "", flags=re.IGNORECASE)
        or re.search(r"\b(?:email|phone|linkedin|twitter)\s*:", text or "", flags=re.IGNORECASE)
        or re.search(r"\bX{2,}(?:[) -]+X{2,})+\b", text or "")
    )


def _looks_like_network_literal(text: str) -> bool:
    """Detect whether text looks like network literal."""
    return bool(
        re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text or "")
        or re.search(r"\bipv[46]\b", text or "", flags=re.IGNORECASE)
    )


def _looks_like_keyboard_shortcut_literal(text: str) -> bool:
    """Detect whether text looks like keyboard shortcut literal."""
    return bool(
        re.search(
            r"`[^`]+`\s*\+\s*`[^`]+`|"
            r"\b(?:ctrl|control|alt|shift|cmd|command|option|esc|enter|tab)\b\s*\+\s*"
            r"\b(?:ctrl|control|alt|shift|cmd|command|option|esc|enter|tab|[a-z0-9])\b",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def _looks_like_spreadsheet_cell_reference(text: str) -> bool:
    """Detect whether text looks like spreadsheet cell reference."""
    return bool(
        re.search(r"\b[A-Z]{1,3}\d+\b(?:\s*(?::|to|-)\s*\b[A-Z]{1,3}\d+\b)?", text or "")
        and re.search(r"\b(?:cell|cells|spreadsheet|excel|sheets?|row|column)\b", text or "", flags=re.IGNORECASE)
    )


def _looks_like_filename_literal(text: str) -> bool:
    """Detect whether text looks like filename literal."""
    return bool(re.search(r"\b[\w.-]+\.(?:csv|json|txt|xml|html|js|py|java|sql|md)\b", text or "", flags=re.IGNORECASE))


def _is_structured_literal_output_task(instruction: str, output_text: str) -> bool:
    """Return whether the input matches structured literal output task."""
    ins = instruction.lower()
    if not _looks_like_structured_literal(output_text):
        return False
    return bool(
        re.search(
            r"\b(table|dataset|data set|kanban|schema|records?|columns?|comparison|report|statement|signature|cv|resume|template)\b",
            ins,
        )
    )


def _is_code_identifier_reference_task(instruction: str, output_text: str) -> bool:
    """Return whether the input matches code identifier reference task."""
    text = f"{instruction}\n{output_text}"
    return bool(
        re.search(r"\b(?:javascript|java|python|code|program|keyword|variable|function)\b", text, flags=re.IGNORECASE)
        and re.search(r"`[^`]+`|'[^']+'|\"[^\"]+\"|“[^”]+”|‘[^’]+’", text)
    )


def _is_exact_quoted_string_task(instruction: str) -> bool:
    """Return whether the input matches exact quoted string task."""
    return bool(
        re.search(
            r"\b(?:length|count|find|compare|reverse|split|palindrome|substring)\b.{0,80}\b(?:string|text)\b",
            instruction,
            flags=re.IGNORECASE,
        )
        and re.search(r"`[^`]+`|'[^']+'|\"[^\"]+\"|“[^”]+”|‘[^’]+’", instruction)
    )


def _instruction_contains_inline_code_payload(instruction: str, input_text: str) -> bool:
    """Handle instruction contains inline code payload."""
    if input_text.strip():
        return False
    return bool(
        re.search(
            r"\b(?:following|piece of|snippet|given|this)\s+code\b.{0,40}[:：]\s*\S|"
            r"\bfunction\s*\([^)]*\)\s*\{",
            instruction,
            flags=re.IGNORECASE,
        )
        and (
            _looks_like_code(instruction)
            or re.search(r"\b(?:int|float|double|string|char|bool|boolean)\s+\w+\b", instruction)
            or re.search(r"\b\w+\s*=\s*[^;\n]+;", instruction)
            or re.search(r"\bfunction\s*\([^)]*\)\s*\{", instruction)
        )
    )


def _looks_like_latex_literal(text: str) -> bool:
    """Detect whether text looks like latex literal."""
    return bool(
        re.search(r"\\\[(?:.|\n)*?\\\]|\\begin\{(?:array|matrix|align|equation)\}", text or "")
        or re.search(r"\\(?:frac|sqrt|text|begin|end)\b", text or "")
    )


def _looks_like_structured_literal(text: str) -> bool:
    """Detect whether text looks like structured literal."""
    t = text.strip()
    if not t:
        return False
    return bool(
        re.search(r"^\s*(?:list|array|string\s*\d*|table|columns?|object)\s*:", t, flags=re.IGNORECASE | re.MULTILINE)
        or re.search(r"^\s*\|.+\|\s*$", t, flags=re.MULTILINE)
        or re.search(r"^[^\n|]+\|[^\n|]+\|[^\n|]+\n\s*:?-{3,}:?\s*\|", t, flags=re.MULTILINE)
        or len(re.findall(r"^[^\n|]+\|[^\n|]+(?:\|[^\n|]+)+$", t, flags=re.MULTILINE)) >= 2
        or re.search(r"^\s*[\[{].*[\]}]\s*$", t, flags=re.DOTALL)
        or re.search(r"\b\w+\s*=\s*(?:\[[^\]]*\]|['\"].*?['\"]|\d+)", t)
        or re.search(r"\b[a-z-]+\s*:\s*[^;\n]+(?:;|$)", t, flags=re.IGNORECASE)
        or re.search(r"^\s*[01]{8}(?:\s+[01]{8})+\s*$", t)
        or re.search(r"\b\d+(?:\s*,\s*\d+)+\b", t)
    )


def _looks_like_url_or_markup_task(instruction: str, input_text: str, output_text: str) -> bool:
    """Detect whether text looks like url or markup task."""
    text = f"{instruction}\n{input_text}\n{output_text}"
    return bool(
        re.search(r"https?://|www\.|<a\s+href=|markdown", text, flags=re.IGNORECASE)
        and re.search(r"\b(link|url|article|html|markdown)\b", instruction, flags=re.IGNORECASE)
    )


def _is_acronym_reference_task(instruction: str) -> bool:
    """Return whether the input matches acronym reference task."""
    return bool(
        re.search(r"\bacronym\b", instruction, flags=re.IGNORECASE)
        and re.search(r"\b(stands?\s+for|meanings?|expansions?)\b", instruction, flags=re.IGNORECASE)
    )


def _is_internet_slang_reference_task(instruction: str, output_text: str) -> bool:
    """Return whether the input matches internet slang reference task."""
    return bool(
        re.search(r"\binternet\s+slang\b|\bslang terms?\b", instruction, flags=re.IGNORECASE)
        and re.search(r"\b(?:DM|FOMO|BRB|NSFW|IRL|RT)\b", output_text or "")
    )


def _is_exact_replacement_literal_task(instruction: str) -> bool:
    """Return whether the input matches exact replacement literal task."""
    return bool(
        re.search(r"\breplace\s+all\s+references?\b", instruction, flags=re.IGNORECASE)
        and re.search(r"`[^`]+`|'[^']+'|\"[^\"]+\"|“[^”]+”|‘[^’]+’", instruction)
    )


def _is_nickname_from_name_task(instruction: str) -> bool:
    """Return whether the input matches nickname from name task."""
    return bool(
        re.search(r"\bnickname\b", instruction, flags=re.IGNORECASE)
        and re.search(r"\bgiven name\b|\bfollowing person\b|\bname\b", instruction, flags=re.IGNORECASE)
    )


def _is_quoted_name_or_title_task(instruction: str, input_text: str, output_text: str) -> bool:
    """Return whether the input matches quoted name or title task."""
    text = f"{instruction}\n{input_text}\n{output_text}"
    return bool(
        re.search(r"`[^`]+`|'[^']+'|\"[^\"]+\"|“[^”]+”|‘[^’]+’", text)
        and re.search(r"\b(?:book|novel|title|game|application|app|product|movie|author)\b", text, flags=re.IGNORECASE)
    )


def _is_dataset_column_literal_task(instruction: str, input_text: str, output_text: str) -> bool:
    """Return whether the input matches dataset column literal task."""
    text = f"{instruction}\n{input_text}\n{output_text}"
    return bool(
        re.search(r"\bcolumns?\b", text, flags=re.IGNORECASE)
        and re.search(r"`[^`]+`|'[^']+'|\"[^\"]+\"|“[^”]+”|‘[^’]+’", text)
    )


def _is_command_line_literal_task(instruction: str, input_text: str) -> bool:
    """Return whether the input matches command line literal task."""
    return bool(
        re.search(r"\bcommand[- ]line|command line|shell command|terminal command\b", instruction, flags=re.IGNORECASE)
    )


def _is_program_output_message_task(instruction: str) -> bool:
    """Return whether the input matches program output message task."""
    return bool(
        re.search(
            r"\b(?:build|write|create|generate)\b.{0,60}\b(?:program|script|code)\b.{0,60}\boutput\s+the\s+following\s+message\b",
            instruction,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\boutput\s+the\s+following\s+message\b.{0,60}\b(?:program|script|code)\b",
            instruction,
            flags=re.IGNORECASE,
        )
    )


def _is_filename_generation_task(instruction: str) -> bool:
    """Return whether the input matches filename generation task."""
    return bool(re.search(r"\b(generate|create|suggest)\b.{0,40}\bfile\s*name|filename\b", instruction, flags=re.IGNORECASE))


def _is_string_word_operation_task(instruction: str) -> bool:
    """Return whether the input matches string word operation task."""
    return bool(
        re.search(
            r"\b(longest substring|common substring|sort the words|split the following string|split .*string|reverse the following words|alphabetical order|morse code|binary string to text)\b",
            instruction,
            flags=re.IGNORECASE,
        )
    )


def _is_alphabetical_word_order_payload_task(instruction: str) -> bool:
    """Return whether the input matches alphabetical word order payload task."""
    return bool(
        re.search(r"\bwords?\b", instruction, flags=re.IGNORECASE)
        and re.search(r"\border\b.{0,80}\balphabet\b|\balphabet\b.{0,80}\border\b", instruction, flags=re.IGNORECASE)
    )


def _is_mask_or_embedded_sentence_transform_task(instruction: str) -> bool:
    """Return whether the input matches mask or embedded sentence transform task."""
    return bool(
        re.search(r"<mask>|\bmask token\b", instruction, flags=re.IGNORECASE)
        or (
            re.search(r"\bmeaning\b.{0,50}\breversed\b|\breversed\b.{0,50}\bmeaning\b", instruction, flags=re.IGNORECASE)
            and re.search(r"\bsentence\s*:", instruction, flags=re.IGNORECASE)
        )
    )




def _is_formula_equation_literal_task(instruction: str, input_text: str) -> bool:
    """Return whether the input matches formula equation literal task."""
    text = f"{instruction}\n{input_text}"
    return bool(
        re.search(r"\b(boolean expression|logical statement|expression|equation|formula|polynomial|coordinates?|radians?|degrees?|math(?:ematical)? query)\b", text, flags=re.IGNORECASE)
        or re.search(r"\b\d+(?:\s*[+\-*/÷x]\s*\d+){1,}\b", text, flags=re.IGNORECASE)
        or re.search(r"\b\d+\s*(?:gb|mb|kb|bytes?|mph|m/s|km/h|kg|g|m|cm|mm|seconds?|minutes?|hours?)\b", text, flags=re.IGNORECASE)
        or re.search(r"[A-Za-z]\s*[+\-*/=<>]\s*\d|\d\s*[+\-*/=<>]\s*[A-Za-z]", input_text)
        or re.search(r"\d+\s*(?:==|!=|<=|>=|<|>)\s*\d+", input_text)
        or re.search(r"\b[a-z]\^\d+\b", input_text, flags=re.IGNORECASE)
        or re.search(r"[π√θ]|\\(?:pi|theta|sqrt)\b", input_text)
    )


def _is_structured_extraction_task(instruction: str, input_text: str, output_text: str) -> bool:
    """Return whether the input matches structured extraction task."""
    return bool(
        re.search(r"\b(extract|link|table|summari[sz]es?|schema|database schema|valid ip addresses?|website)\b", instruction, flags=re.IGNORECASE)
        and (
            _looks_like_structured_literal(input_text)
            or _looks_like_structured_literal(output_text)
            or re.search(r"\bdatabase schema\b", instruction, flags=re.IGNORECASE)
            or re.search(r"https?://|www\.|\b\d{1,3}(?:\.\d{1,3}){3}\b", f"{input_text}\n{output_text}")
        )
    )


def _has_classification_label_set(instruction: str) -> bool:
    """Return whether the input has classification label set."""
    ins = instruction.lower()
    return bool(
        re.search(r"\bas\s+(?:incorrect|correct)\s+or\s+(?:incorrect|correct)\b", ins)
        or re.search(r"\b(?:incorrect|correct)\s+or\s+(?:incorrect|correct)\b", ins)
        or re.search(r"\bas\s+(?:noun|verb|adjective|adverb|other)\b", ins)
        or re.search(r"\bclassify\b.{0,80}\b(?:noun|verb|adjective|adverb|other)\b", ins)
    )


def apply_field_guardrails(
    sample: Dict,
    target_langs: List[str],
    decisions_by_lang: Dict[str, Dict[str, Dict[str, object]]],
    policy_by_field: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], Dict[str, Dict[str, object]]]:
    """Apply field guardrails."""
    # Start from Gemma's field policy and enforce task-specific constraints
    # (classification labels, translation payload locking, code/literals, etc.).
    task = str(sample.get("primary_task", "")).strip().lower()
    field_text = {
        field: _as_text(sample.get(field, ""))
        for field in POLICY_FIELDS
    }
    instruction = field_text["instruction"]
    input_text = field_text["input"]
    output_text = field_text["output"]

    mutator = FieldPolicyMutator(
        target_langs=target_langs,
        decisions_by_lang=decisions_by_lang,
        policy_by_field=policy_by_field,
        looks_like_code=_looks_like_code,
    )

    force_field_false = mutator.force_field_false
    force_field_false_for_langs = mutator.force_field_false_for_langs
    maybe_relax_code_content_false = mutator.maybe_relax_code_content_false
    
    # If incoming policy already marked text as code-content but the new
    # detector disagrees, relax back to translatable before applying guardrails.
    maybe_relax_code_content_false("input", input_text)
    maybe_relax_code_content_false("output", output_text)

    # For translation tasks, keep source/target payload fields fixed.
    if task == "translation":
        target_language_tags = _find_translation_target_language_tags(instruction)
        force_field_false("output", "translation target text")
        if input_text.strip():
            if _translation_has_explicit_source_language(instruction):
                force_field_false("input", "translation explicit source")
            else:
                force_field_false_for_langs("input", target_language_tags, "translation target language")
        else:
            force_field_false_for_langs(
                "instruction",
                target_language_tags,
                "translation target language",
            )
    elif task != "question_answering" and _looks_like_direct_translation_prompt(instruction):
        force_field_false("input", "translation prompt source")
        force_field_false("output", "translation prompt target")

    if _is_generated_identifier_task(instruction):
        force_field_false("output", "generated identifier")
        if _identifier_input_should_be_preserved(input_text) or (
            input_text.strip()
            and re.search(r"\b(given|input|provided|based on|with the following phrase|following phrase)\b", instruction, flags=re.IGNORECASE)
        ):
            force_field_false("input", "identifier source data")

    if _is_regex_task(instruction):
        if input_text.strip():
            force_field_false("input", "regex source pattern")
        force_field_false("output", "regex pattern")

    if _is_cipher_or_literal_transform_task(instruction):
        if input_text.strip():
            force_field_false("input", "literal source text")
        force_field_false("output", "literal transformed text")

    if _is_table_json_task(instruction):
        if input_text.strip():
            force_field_false("input", "structured source data")
        force_field_false("output", "structured output")

    if _is_literal_transform_task(instruction):
        if input_text.strip():
            force_field_false("input", "literal source text")
        force_field_false("output", "literal transformed text")

    if _is_list_or_array_transform_task(instruction, input_text):
        if input_text.strip():
            force_field_false("input", "structured source data")
        force_field_false("output", "structured output")

    if _is_code_edit_task(instruction):
        if input_text.strip():
            force_field_false("input", "code content")
        if _looks_like_code(output_text) or task == "coding":
            force_field_false("output", "code content")

    if task == "coding":
        if _looks_like_structured_literal(input_text):
            force_field_false("input", "structured source data")
        if _looks_like_structured_literal(output_text):
            force_field_false("output", "structured output")

    if _looks_like_url_or_markup_task(instruction, input_text, output_text):
        if input_text.strip():
            force_field_false("input", "url or markup payload")
        force_field_false("output", "url or markup payload")

    if re.search(r"https?://|www\.", instruction, flags=re.IGNORECASE):
        force_field_false("instruction", "url or markup payload")

    if _is_acronym_reference_task(instruction):
        force_field_false("output", "fixed acronym expansion")

    if _is_internet_slang_reference_task(instruction, output_text):
        force_field_false("output", "fixed slang/acronym tokens")

    if _is_exact_replacement_literal_task(instruction):
        force_field_false("instruction", "exact replacement literal")
        if input_text.strip():
            force_field_false("input", "exact replacement source")
        force_field_false("output", "exact replacement output")

    if _is_nickname_from_name_task(instruction):
        if input_text.strip():
            force_field_false("input", "name literal")
        force_field_false("output", "generated name literal")

    if _is_quoted_name_or_title_task(instruction, input_text, output_text):
        if input_text.strip():
            force_field_false("input", "name/title literal")
        force_field_false("output", "name/title literal")

    if _is_dataset_column_literal_task(instruction, input_text, output_text):
        force_field_false("instruction", "dataset column literal")
        if input_text.strip():
            force_field_false("input", "dataset column literal")
        force_field_false("output", "dataset column literal")

    if _is_command_line_literal_task(instruction, input_text):
        if input_text.strip():
            force_field_false("input", "command-line literal")
        force_field_false("output", "command-line literal")

    if _is_program_output_message_task(instruction):
        if input_text.strip():
            force_field_false("input", "program output message")
        force_field_false("output", "code content")

    if _is_filename_generation_task(instruction):
        force_field_false("output", "filename literal")

    if _looks_like_filename_literal(input_text):
        force_field_false("input", "filename literal")
    if _looks_like_filename_literal(output_text):
        force_field_false("output", "filename literal")

    if _is_string_word_operation_task(instruction):
        if input_text.strip():
            force_field_false("input", "literal source text")
        else:
            force_field_false("instruction", "literal source text")
        force_field_false("output", "literal transformed text")

    if _is_alphabetical_word_order_payload_task(instruction):
        if input_text.strip():
            force_field_false("input", "literal source text")
        force_field_false("output", "literal transformed text")

    if _is_mask_or_embedded_sentence_transform_task(instruction):
        force_field_false("instruction", "literal source text")
        force_field_false("output", "literal transformed text")

    if _is_search_or_catalog_alphabetical_task(instruction):
        force_field_false("output", "ordered list output")

    if _is_formula_equation_literal_task(instruction, input_text):
        if input_text.strip():
            force_field_false("input", "formula or equation")
        force_field_false("output", "formula or equation")

    if _is_structured_extraction_task(instruction, input_text, output_text):
        if input_text.strip():
            force_field_false("input", "structured source data")
        force_field_false("output", "structured output")

    if _looks_like_structured_literal(input_text):
        force_field_false("input", "structured source data")
    if _is_structured_literal_output_task(instruction, output_text):
        force_field_false("output", "structured output")

    if _looks_like_markup_literal(input_text):
        force_field_false("input", "url or markup payload")
    if _looks_like_markup_literal(output_text):
        force_field_false("output", "url or markup payload")

    if _looks_like_placeholder_literal(input_text):
        force_field_false("input", "placeholder literal")
    if _looks_like_placeholder_literal(output_text):
        force_field_false("output", "placeholder literal")

    if _looks_like_hashtag_literal(input_text):
        force_field_false("input", "hashtag literal")
    if _looks_like_hashtag_literal(output_text):
        force_field_false("output", "hashtag literal")

    if _looks_like_music_chord_sequence(input_text):
        force_field_false("input", "music chord sequence")
    if _looks_like_music_chord_sequence(output_text):
        force_field_false("output", "music chord sequence")

    if _looks_like_boolean_expression_literal(input_text):
        force_field_false("input", "boolean expression")
    if _looks_like_boolean_expression_literal(output_text):
        force_field_false("output", "boolean expression")

    if _looks_like_binary_literal(input_text):
        force_field_false("input", "binary literal")
    if _looks_like_binary_literal(output_text):
        force_field_false("output", "binary literal")

    if _looks_like_command_invocation(input_text):
        force_field_false("input", "command-line literal")
    if _looks_like_command_invocation(output_text):
        force_field_false("output", "command-line literal")

    if _looks_like_filesystem_path_literal(input_text):
        force_field_false("input", "path literal")
    if _looks_like_filesystem_path_literal(output_text):
        force_field_false("output", "path literal")

    if _looks_like_compact_math_literal(output_text) and re.search(
        r"\b(?:factor|factorial|fraction|equation|expression|solve|calculate|compute|evaluate)\b",
        instruction,
        flags=re.IGNORECASE,
    ):
        force_field_false("output", "formula or equation")

    if _looks_like_player_label_literal(input_text):
        force_field_false("input", "player label literal")
    if _looks_like_player_label_literal(output_text):
        force_field_false("output", "player label literal")

    if _looks_like_hex_color_literal(input_text):
        force_field_false("input", "hex color literal")
    if _looks_like_hex_color_literal(output_text):
        force_field_false("output", "hex color literal")

    if _looks_like_dna_sequence_literal(input_text):
        force_field_false("input", "generated biological sequence")
    if _looks_like_dna_sequence_literal(output_text):
        force_field_false("output", "generated biological sequence")

    if _looks_like_objective_label_literal(input_text):
        force_field_false("input", "handle and fixed label literal")
    if _looks_like_objective_label_literal(output_text):
        force_field_false("output", "handle and fixed label literal")

    if _looks_like_foreign_phrase_payload(input_text):
        force_field_false("input", "foreign phrase payload")
    if _looks_like_foreign_phrase_payload(output_text):
        force_field_false("output", "foreign phrase payload")

    if _looks_like_url_or_contact_literal(input_text):
        force_field_false("input", "url or contact literal")
    if _looks_like_url_or_contact_literal(output_text):
        force_field_false("output", "url or contact literal")

    if _looks_like_network_literal(input_text):
        force_field_false("input", "network literal")
    if _looks_like_network_literal(output_text):
        force_field_false("output", "network literal")

    if _looks_like_keyboard_shortcut_literal(input_text):
        force_field_false("input", "keyboard shortcut")
    if _looks_like_keyboard_shortcut_literal(output_text):
        force_field_false("output", "keyboard shortcut")

    if _looks_like_spreadsheet_cell_reference(input_text):
        force_field_false("input", "spreadsheet cell reference")
    if _looks_like_spreadsheet_cell_reference(output_text):
        force_field_false("output", "spreadsheet cell reference")

    if _is_exact_quoted_string_task(instruction):
        force_field_false("instruction", "exact string literal")
        if output_text.strip() and _is_short_output_text(output_text):
            force_field_false("output", "exact string result")

    if _instruction_contains_inline_code_payload(instruction, input_text):
        force_field_false("instruction", "inline code payload")

    if _looks_like_latex_literal(input_text):
        force_field_false("input", "latex source")
    if _looks_like_latex_literal(output_text):
        force_field_false("output", "latex source")

    if _is_code_identifier_reference_task(instruction, output_text):
        force_field_false("output", "code identifier")

    if task == "classification" and _has_classification_label_set(instruction):
        force_field_false("instruction", "classification canonical labels")
        force_field_false("output", "classification fixed labels")

    if task == "classification":
        classification_input_reason = _classification_input_protect_reason(instruction)
        if classification_input_reason:
            force_field_false("input", classification_input_reason)
        classification_output_reason = _classification_output_protect_reason(instruction, output_text)
        if classification_output_reason:
            force_field_false("output", classification_output_reason)
            force_field_false("instruction", "classification canonical labels")

    if task == "coding":
        coding_input_reason = _coding_input_protect_reason(instruction, input_text)
        if coding_input_reason:
            force_field_false("input", coding_input_reason)
        coding_output_reason = _coding_output_protect_reason(instruction, input_text, output_text)
        if coding_output_reason:
            force_field_false("output", coding_output_reason)

    output_reason_by_task = {
        "evaluation": _evaluation_output_protect_reason(instruction, output_text),
        "other": _other_output_protect_reason(instruction, output_text),
        "question_answering": _question_answering_output_protect_reason(instruction, input_text),
    }
    output_reason = output_reason_by_task.get(task, "") or _fixed_language_output_protect_reason(
        instruction,
        output_text,
    )
    if output_reason:
        force_field_false("output", output_reason)

    # Protect executable/code-like content.
    if _looks_like_code(input_text):
        force_field_false("input", "code content")
    if _looks_like_code(output_text):
        force_field_false("output", "code content")

    # Protect tightly constrained label spaces.
    if task in {"evaluation", "classification"} and _is_constrained_label_output(output_text):
        force_field_false("output", "fixed label set")

    return decisions_by_lang, policy_by_field

__all__ = [
    "apply_field_guardrails",
    "_looks_like_structured_literal",
    "_is_formula_equation_literal_task",
]
