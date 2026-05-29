"""Policy normalization helpers."""

from typing import Dict, List, Optional, Tuple

from .constants import POLICY_FIELDS
from .utils import _brief_reason, _compact_text, _normalize_sample_reason_code

def _sample_policy_metadata(
    include_sample: bool,
    reason_code: object = "",
    evidence: object = "",
    source: str = "gemma",
) -> Dict[str, object]:
    """Handle sample policy metadata."""
    normalized_evidence = ""
    if not include_sample and isinstance(evidence, str):
        normalized_evidence = _compact_text(evidence)
    normalized_source = source if isinstance(source, str) and source.strip() else "gemma"
    return {
        "include_sample": include_sample,
        "reason_code": _normalize_sample_reason_code(reason_code, include_sample),
        "evidence": normalized_evidence,
        "source": normalized_source.strip(),
    }


def _field_decision_obj(should_translate: bool, reason: str = "") -> Dict[str, object]:
    """Build a compact field decision object."""
    # Keep reasons only for negative decisions to keep policy compact/noise-free.
    obj: Dict[str, object] = {"should_translate": should_translate}
    if not should_translate:
        obj["reason"] = _brief_reason(reason, "no reason")
    return obj

def normalize_decisions_multilang(
    parsed: Optional[dict],
    sample: Dict,
    target_langs: List[str],
    default_should_translate: bool,
) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], Dict[str, Dict[str, object]]]:
    """Normalize decisions multilang."""
    # Build per-language decisions from model JSON, then compact into canonical
    # policy form: global default + per-language false exceptions.
    # Note: default_should_translate is a fallback path when parsed JSON is
    # missing/invalid.
    result_by_lang: Dict[str, Dict[str, Dict[str, object]]] = {}
    for lang in target_langs:
        result_by_lang[lang.lower()] = {}

    policy_by_field: Dict[str, Dict[str, object]] = {}

    for field in POLICY_FIELDS:
        value = sample.get(field, "")
        # Empty fields are never translated regardless of model output.
        if not isinstance(value, str) or not value.strip():
            policy_by_field[field] = {
                "all_languages": _field_decision_obj(False, "empty_or_missing_field"),
                "per_language": {},
            }
            for lang in result_by_lang:
                result_by_lang[lang][field] = _field_decision_obj(False, "empty_or_missing_field")
            continue

        default_decision = default_should_translate
        default_reason = "no reason"
        per_language_overrides: Dict[str, Dict[str, object]] = {}

        field_obj = parsed.get(field) if isinstance(parsed, dict) else None
        if isinstance(field_obj, dict):
            all_obj = field_obj.get("all_languages")
            if isinstance(all_obj, dict):
                decision = all_obj.get("should_translate")
                reason = all_obj.get("reason")
                if isinstance(decision, bool):
                    default_decision = decision
                    if not default_decision:
                        default_reason = _brief_reason(reason, "no reason")
                else:
                    if not default_decision:
                        default_reason = "invalid decision"
            else:
                # Backward-compatible support for older single-decision format.
                decision = field_obj.get("should_translate")
                reason = field_obj.get("reason")
                if isinstance(decision, bool):
                    default_decision = decision
                    if not default_decision:
                        default_reason = _brief_reason(reason, "no reason")
                else:
                    if not default_decision:
                        default_reason = "invalid decision"

            per_lang_obj = field_obj.get("per_language")
            if isinstance(per_lang_obj, dict):
                for lang_key, lang_decision_obj in per_lang_obj.items():
                    if not isinstance(lang_key, str):
                        continue
                    lang = lang_key.lower()
                    if lang not in result_by_lang:
                        continue
                    if not isinstance(lang_decision_obj, dict):
                        continue
                    d = lang_decision_obj.get("should_translate")
                    r = lang_decision_obj.get("reason")
                    if not isinstance(d, bool):
                        continue
                    per_language_overrides[lang] = _field_decision_obj(
                        d,
                        _brief_reason(r, "no reason"),
                    )

        # Build final per-language decisions first.
        final_by_lang: Dict[str, Dict[str, object]] = {}
        for lang in result_by_lang:
            if lang in per_language_overrides:
                final_by_lang[lang] = dict(per_language_overrides[lang])
            else:
                final_by_lang[lang] = _field_decision_obj(default_decision, default_reason)
            result_by_lang[lang][field] = dict(final_by_lang[lang])

        # Canonical compact policy:
        # - all_languages=true + per_language only false exceptions, OR
        # - all_languages=false (all languages blocked).
        # This keeps output small and easy to audit while preserving exact
        # final behavior for every target language.
        false_langs = [
            lang
            for lang, obj in final_by_lang.items()
            if not bool(obj.get("should_translate"))
        ]
        if final_by_lang and len(false_langs) == len(final_by_lang):
            reason = default_reason
            if (not isinstance(reason, str) or not reason.strip()) and false_langs:
                reason = _brief_reason(final_by_lang[false_langs[0]].get("reason"), "no reason")
            policy_by_field[field] = {
                "all_languages": _field_decision_obj(False, reason),
                "per_language": {},
            }
        else:
            per_language_false_overrides: Dict[str, Dict[str, object]] = {}
            for lang in false_langs:
                r = _brief_reason(final_by_lang[lang].get("reason"), "no reason")
                per_language_false_overrides[lang] = _field_decision_obj(False, r)
            policy_by_field[field] = {
                "all_languages": _field_decision_obj(True, ""),
                "per_language": per_language_false_overrides,
            }

    return result_by_lang, policy_by_field


def normalize_sample_policy(
    parsed: Optional[dict],
    default_include_sample: bool,
) -> Dict[str, object]:
    """Normalize sample policy."""
    # Normalize sample-level gate. If parsed JSON is missing/invalid, fallback is
    # controlled by default_include_sample. The sample decision is intentionally
    # global; per-language exceptions belong to the field-level policy.
    metadata = _sample_policy_metadata(
        include_sample=default_include_sample,
        reason_code="directly_translatable" if default_include_sample else "other_not_directly_translatable",
        evidence="",
        source="fallback",
    )

    # parsed may be None (failed JSON extraction) or malformed.
    # In those cases we preserve fallback defaults initialized above.
    sample_obj = parsed.get("sample") if isinstance(parsed, dict) else None
    if isinstance(sample_obj, dict):
        # New format: global sample gate.
        # Expected shape:
        # {
        #   "sample": {
        #     "include_sample": <bool>,
        #     "reason_code": <enum>,
        #     "evidence": <short quote>
        #   }
        # }
        d = sample_obj.get("include_sample")
        if isinstance(d, bool):
            # Valid model decision replaces fallback default.
            metadata = _sample_policy_metadata(
                include_sample=d,
                reason_code=sample_obj.get("reason_code"),
                evidence=sample_obj.get("evidence"),
                source=sample_obj.get("source") or "gemma",
            )
        else:
            # Backward-compatible support for old all_languages/per_language shape.
            # Legacy shape expected:
            # {
            #   "sample": {
            #     "all_languages": {"include_sample": <bool>, "reason": <optional str>}
            #   }
            # }
            all_obj = sample_obj.get("all_languages")
            if isinstance(all_obj, dict):
                d2 = all_obj.get("include_sample")
                if isinstance(d2, bool):
                    metadata = _sample_policy_metadata(
                        include_sample=d2,
                        reason_code=all_obj.get("reason"),
                        evidence="",
                        source=sample_obj.get("source") or "gemma",
                    )

    # include_sample drives the pipeline gate:
    # - False -> skip field-level policy inference entirely
    # - True  -> continue to field-level policy stage
    return metadata

__all__ = [
    "_sample_policy_metadata",
    "_field_decision_obj",
    "normalize_decisions_multilang",
    "normalize_sample_policy",
]
