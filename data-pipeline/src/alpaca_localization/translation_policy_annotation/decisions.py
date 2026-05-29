"""Sample and field decision orchestration."""

import argparse
from typing import Dict, List, Tuple

from .constants import POLICY_FIELDS
from .field_guardrails import apply_field_guardrails
from .gemma_client import GemmaJSONParseError, query_gemma_json
from .io_utils import quarantine_parse_failure
from .normalize import normalize_decisions_multilang, normalize_sample_policy, _sample_policy_metadata
from .prompts import build_field_policy_prompt, build_sample_policy_prompt
from .sample_guardrails import apply_sample_guardrails

def decide_sample_policy(
    sample: Dict,
    args: argparse.Namespace,
    source_lang_name: str,
    target_langs: List[str],
) -> Tuple[Dict[str, object], bool]:
    """Run pass 1 and report whether sample-level JSON failed to parse.

    This is intentionally separate from field decisions. The returned metadata
    contains the final sample gate plus reason/evidence/source audit fields.
    """
    if args.reuse_existing_policies_without_querying_gemma:
        existing_sample_policy = sample.get(args.output_sample_policy_field)
        sample_parsed = (
            {
                "sample": {
                    "include_sample": existing_sample_policy,
                    "reason_code": sample.get(args.output_sample_policy_reason_field),
                    "evidence": sample.get(args.output_sample_policy_evidence_field),
                    "source": sample.get(args.output_sample_policy_source_field) or "existing_policy",
                }
            }
            if isinstance(existing_sample_policy, bool)
            else None
        )
    else:
        sample_prompt = build_sample_policy_prompt(
            sample=sample,
            source_lang_name=source_lang_name,
            target_langs=target_langs,
        )
        try:
            sample_parsed = query_gemma_json(sample_prompt, args)
        except GemmaJSONParseError as e:
            quarantine_parse_failure(
                sample=sample,
                args=args,
                stage="sample_policy",
                prompt=sample_prompt,
                raw_response=e.raw_response,
            )
            # Conservative fallback: do not localize samples whose sample gate
            # could not be parsed.
            return _sample_policy_metadata(
                include_sample=False,
                reason_code="other_not_directly_translatable",
                evidence="sample policy JSON parse failure",
                source="sample_policy_parse_failure",
            ), True

    # Missing/unclear policy keys are saved for audit and normalized through the
    # conservative fallback below. Unparseable JSON is handled above by
    # quarantining the sample and returning a parse-failure decision.

    sample_policy = normalize_sample_policy(
        parsed=sample_parsed,
        default_include_sample=False,
    )
    sample_policy = apply_sample_guardrails(
        sample=sample,
        sample_policy=sample_policy,
    )
    return sample_policy, False


def decide_field_policy(
    sample: Dict,
    args: argparse.Namespace,
    source_lang_name: str,
    target_langs: List[str],
    sample_policy: bool,
) -> Tuple[Dict[str, Dict[str, object]], bool]:
    """Run pass 2 and report whether field annotation failed to parse."""
    if not sample_policy:
        return {}, False

    if args.reuse_existing_policies_without_querying_gemma:
        field_parsed = sample.get(args.output_policy_field)
    else:
        field_prompt = build_field_policy_prompt(
            sample=sample,
            source_lang_name=source_lang_name,
            target_langs=target_langs,
            sample_policy=sample_policy,
        )
        try:
            field_parsed = query_gemma_json(field_prompt, args)
        except GemmaJSONParseError as e:
            quarantine_parse_failure(
                sample=sample,
                args=args,
                stage="field_policy",
                prompt=field_prompt,
                raw_response=e.raw_response,
            )
            # Conservative fallback: if field policy is not parseable, do not
            # localize this sample.
            return {}, True

    decisions_by_lang, policy_by_field = normalize_decisions_multilang(
        parsed=field_parsed,
        sample=sample,
        target_langs=target_langs,
        default_should_translate=False,
    )
    _decisions_by_lang, policy_by_field = apply_field_guardrails(
        sample=sample,
        target_langs=target_langs,
        decisions_by_lang=decisions_by_lang,
        policy_by_field=policy_by_field,
    )
    return policy_by_field, False


def annotate_sample(
    sample: Dict,
    args: argparse.Namespace,
    source_lang_name: str,
    target_langs: List[str],
) -> Tuple[Dict, bool, int]:
    """Annotate one row and report skip/quarantine counters for summaries."""
    sample_policy, sample_parse_failed = decide_sample_policy(
        sample=sample,
        args=args,
        source_lang_name=source_lang_name,
        target_langs=target_langs,
    )
    include_sample = bool(sample_policy.get("include_sample"))
    policy_by_field, field_parse_failed = decide_field_policy(
        sample=sample,
        args=args,
        source_lang_name=source_lang_name,
        target_langs=target_langs,
        sample_policy=include_sample,
    )
    if field_parse_failed:
        sample_policy = _sample_policy_metadata(
            include_sample=False,
            reason_code="other_not_directly_translatable",
            evidence="field policy JSON parse failure",
            source="field_policy_parse_failure",
        )
        include_sample = False

    enriched = dict(sample)
    enriched[args.output_sample_policy_field] = include_sample
    enriched[args.output_sample_policy_reason_field] = sample_policy["reason_code"]
    evidence = sample_policy.get("evidence")
    # If evidence is non-empty, write it.
    # If evidence is empty or missing, remove the evidence field from the output if it already existed. 
    if isinstance(evidence, str) and evidence.strip():
        enriched[args.output_sample_policy_evidence_field] = evidence
    else:
        enriched.pop(args.output_sample_policy_evidence_field, None)
    # Writes where the final sample decision came from.
    enriched[args.output_sample_policy_source_field] = sample_policy["source"]
    # If the sample is kept, write the field-level policy
    if include_sample:
        enriched[args.output_policy_field] = policy_by_field
    else:
        enriched[args.output_policy_field] = None
    # Counts Gemma JSON parse failures for this sample
    parse_failure_count = int(sample_parse_failed) + int(field_parse_failed)
    return enriched, not include_sample, parse_failure_count

__all__ = [
    "decide_sample_policy",
    "decide_field_policy",
    "annotate_sample",
]
