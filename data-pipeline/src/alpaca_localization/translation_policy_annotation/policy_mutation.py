"""Helpers for mutating compact field policies.

These helpers keep the compact field policy and the expanded per-language
decisions aligned.
"""

from typing import Callable, Dict, List

from .normalize import _field_decision_obj


class FieldPolicyMutator:
    """Mutate compact field policies while keeping per-language decisions aligned."""

    def __init__(
        self,
        target_langs: List[str],
        decisions_by_lang: Dict[str, Dict[str, Dict[str, object]]],
        policy_by_field: Dict[str, Dict[str, object]],
        looks_like_code: Callable[[str], bool],
    ) -> None:
        self.target_langs = target_langs
        self.decisions_by_lang = decisions_by_lang
        self.policy_by_field = policy_by_field
        self.looks_like_code = looks_like_code

    def force_field_false(self, field: str, reason: str) -> None:
        """Force a field to be non-translatable for all target languages."""
        self.policy_by_field[field] = {
            "all_languages": _field_decision_obj(False, reason),
            "per_language": {},
        }

        for lang in self.target_langs:
            l = lang.lower()
            if l not in self.decisions_by_lang:
                self.decisions_by_lang[l] = {}
            self.decisions_by_lang[l][field] = _field_decision_obj(False, reason)

    def force_field_true(self, field: str) -> None:
        """Force a field to be translatable for all target languages."""
        self.policy_by_field[field] = {
            "all_languages": _field_decision_obj(True, ""),
            "per_language": {},
        }

        for lang in self.target_langs:
            l = lang.lower()
            if l not in self.decisions_by_lang:
                self.decisions_by_lang[l] = {}
            self.decisions_by_lang[l][field] = _field_decision_obj(True, "")

    def force_field_false_for_langs(
        self,
        field: str,
        langs: List[str],
        reason: str,
    ) -> None:
        """Force a field to be non-translatable for specific target languages."""
        if not langs:
            return

        field_policy = self.policy_by_field.get(field)
        if not isinstance(field_policy, dict):
            field_policy = {
                "all_languages": _field_decision_obj(True, ""),
                "per_language": {},
            }
            self.policy_by_field[field] = field_policy

        all_obj = field_policy.get("all_languages")

        # Important behavior:
        # If the whole field is already globally false, do not weaken it into
        # per-language exceptions.
        if isinstance(all_obj, dict) and all_obj.get("should_translate") is False:
            return

        field_policy["all_languages"] = _field_decision_obj(True, "")

        per_language = field_policy.get("per_language")
        if not isinstance(per_language, dict):
            per_language = {}
            field_policy["per_language"] = per_language

        allowed_langs = {lang.lower() for lang in self.target_langs}

        for lang in langs:
            l = lang.lower()
            if l not in allowed_langs:
                continue

            per_language[l] = _field_decision_obj(False, reason)

            if l not in self.decisions_by_lang:
                self.decisions_by_lang[l] = {}
            self.decisions_by_lang[l][field] = _field_decision_obj(False, reason)

    def maybe_relax_code_content_false(self, field: str, value: str) -> None:
        """Relax false code-content decisions when code is not actually detected."""
        if not isinstance(value, str) or not value.strip():
            return

        field_policy = self.policy_by_field.get(field)
        if not isinstance(field_policy, dict):
            return

        all_obj = field_policy.get("all_languages")
        if not isinstance(all_obj, dict):
            return

        code_reasons = {"code content", "code/command should not be translated"}
        reason = str(all_obj.get("reason") or "").strip().lower()

        if all_obj.get("should_translate") is False and reason in code_reasons:
            if not self.looks_like_code(value):
                self.force_field_true(field)
            return

        per_language = field_policy.get("per_language")
        if not isinstance(per_language, dict) or self.looks_like_code(value):
            return

        removed_langs = []

        for lang, obj in list(per_language.items()):
            if not isinstance(obj, dict):
                continue

            lang_reason = str(obj.get("reason") or "").strip().lower()

            if obj.get("should_translate") is False and lang_reason in code_reasons:
                removed_langs.append(lang)
                del per_language[lang]

        for lang in removed_langs:
            l = lang.lower()
            if l not in self.decisions_by_lang:
                self.decisions_by_lang[l] = {}
            self.decisions_by_lang[l][field] = _field_decision_obj(True, "")


__all__ = ["FieldPolicyMutator"]