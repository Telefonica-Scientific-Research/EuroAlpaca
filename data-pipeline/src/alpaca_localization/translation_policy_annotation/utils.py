"""Shared text and regex helpers."""

import re
from typing import Iterable

from .constants import SAMPLE_REASON_CODES

def _brief_reason(raw_reason: object, fallback: str) -> str:
    """Return a compact reason string with a fallback."""
    if isinstance(raw_reason, str):
        compact = " ".join(raw_reason.strip().split())
        if compact:
            return compact[:80]
    return fallback


def _as_text(value: object) -> str:
    """Convert a value to annotation text."""
    return value if isinstance(value, str) else ""


def _matches_any(patterns: Iterable[str], text: str, flags: int = 0) -> bool:
    """Return whether any regex pattern matches text."""
    return any(re.search(pattern, text, flags=flags) for pattern in patterns)


def _reason_if_matches(patterns: Iterable[str], text: str, reason: str, flags: int = 0) -> str:
    """Return a reason when any regex pattern matches text."""
    return reason if _matches_any(patterns, text, flags=flags) else ""


def _compact_text(text: object, limit: int = 160) -> str:
    """Compact whitespace and truncate evidence text."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())[:limit]


def _normalize_sample_reason_code(raw_reason: object, include_sample: bool) -> str:
    """Normalize sample reason code."""
    if include_sample:
        return "directly_translatable"
    if isinstance(raw_reason, str):
        compact = raw_reason.strip()
        if compact in SAMPLE_REASON_CODES:
            return compact
    return "other_not_directly_translatable"

def _word_count(text: str) -> int:
    """Count word-like tokens in text."""
    return len(re.findall(r"\b[\w']+\b", text or ""))

__all__ = [
    "_brief_reason",
    "_as_text",
    "_matches_any",
    "_reason_if_matches",
    "_compact_text",
    "_normalize_sample_reason_code",
    "_word_count",
]
