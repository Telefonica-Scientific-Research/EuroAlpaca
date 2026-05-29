"""Shared task-shape detectors used by sample and field guardrails."""

import re


def _is_generated_identifier_task(instruction: str) -> bool:
    """Return whether the input matches generated identifier task."""
    ins = instruction.lower()
    if re.search(r"\b(tweet|twitter)\b", ins):
        return False
    if re.search(r"\bsentence\b", ins) and re.search(r"\bletter\b|\bverb\b", ins):
        return False
    return bool(
        re.search(r"\b(passwords?|passcodes?|usernames?|user names?|random string|alphanumeric string|encryption key)\b", ins)
        or re.search(r"\b(?:generate|create)\b.{0,60}\b(?:letters?|digits?|characters?)\b", ins)
    )


def _is_list_or_array_transform_task(instruction: str, input_text: str) -> bool:
    """Return whether the input matches list or array transform task."""
    text = f"{instruction}\n{input_text}".lower()
    return bool(
        re.search(r"\b(list|array|matrix|matrices|strings?|numbers?|integers?)\b", text)
        and re.search(r"\b(sort|arrange|rearrange|reverse|merge|reflect|sum|standard deviation|alphabetically|alphabetical)\b", text)
    )


def _is_search_or_catalog_alphabetical_task(instruction: str) -> bool:
    """Return whether the input matches search or catalog alphabetical task."""
    if re.search(r"\b(reorder|sort|arrange|rearrange|order)\b", instruction, flags=re.IGNORECASE):
        return False
    return bool(
        re.search(r"\b(find|list|search|books?|titles?|keyword)\b", instruction, flags=re.IGNORECASE)
        and re.search(r"\balphabetical(?:ly)?\s+order\b|\balphabetically\b", instruction, flags=re.IGNORECASE)
    )


__all__ = [
    "_is_generated_identifier_task",
    "_is_list_or_array_transform_task",
    "_is_search_or_catalog_alphabetical_task",
]
