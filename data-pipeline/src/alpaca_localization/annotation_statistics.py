#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Tuple


DEFAULT_INPUT = "/mnt/mirror1/aleixsant/alpaca_cleaned/en/alpaca_cleaned_labeled_annotated_gemma4.jsonl" #alpaca_cleaned_labeled_annotated_NEW.jsonl"
TEXT_FIELDS = ("instruction", "input", "output")
LIST_LABEL_FIELDS = ("secondary_tasks", "secondary_domains")
SCALAR_LABEL_FIELDS = ("primary_task", "primary_domain")


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object on line {line_no}, got {type(obj).__name__}")
            yield line_no, obj


def empty_counter() -> Counter:
    return Counter({"true": 0, "false": 0, "missing": 0, "invalid": 0})


def bool_bucket(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "missing"
    return "invalid"


def clean_label(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "<missing_or_invalid>"


def iter_clean_list(value: Any) -> Iterable[str]:
    if not isinstance(value, list):
        yield "<missing_or_invalid>"
        return
    if not value:
        yield "<empty>"
        return
    emitted = False
    for item in value:
        if isinstance(item, str) and item.strip():
            emitted = True
            yield item.strip()
    if not emitted:
        yield "<missing_or_invalid>"


def percent(n: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{(100.0 * n / total):.2f}%"


def print_counter(title: str, counter: Counter, total: int = 0, top: int = 0) -> None:
    print(title)
    items = counter.most_common(top) if top > 0 else counter.most_common()
    if not items:
        print("  <none>")
    for key, value in items:
        suffix = f" ({percent(value, total)})" if total else ""
        print(f"  {str(key):40s} {value}{suffix}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print annotation statistics for Alpaca task/domain labels and translation policies."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Annotated JSONL input path.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="If >0, print only the top-N values for open-ended counters.",
    )
    args = parser.parse_args()

    stats: Dict[str, Any] = {
        "total_samples": 0,
        "sample_translation_policy": empty_counter(),
        "sample_translation_policy_reason": Counter(),
        "sample_translation_policy_source": Counter(),
        "text_fields": {
            field: Counter({"blank": 0, "not_blank": 0, "missing_or_invalid": 0})
            for field in TEXT_FIELDS
        },
        "labels": {
            "primary_task": Counter(),
            "secondary_tasks": Counter(),
            "primary_domain": Counter(),
            "secondary_domains": Counter(),
        },
        "field_should_translate": {
            field: empty_counter()
            for field in TEXT_FIELDS
        },
        "field_should_translate_reason": {
            field: Counter()
            for field in TEXT_FIELDS
        },
        "per_language_should_translate": defaultdict(empty_counter),
        "per_language_should_translate_by_field": defaultdict(empty_counter),
        "field_policy_shape": Counter(),
        "field_preservation": Counter({
            "samples_with_at_least_one_field_false": 0,
            "samples_with_at_least_one_nonblank_field_false": 0,
        }),
        "input_field": Counter({
            "blank": 0,
            "not_blank": 0,
            "not_blank_and_sample_true_field_false": 0,
        }),
    }

    for _, sample in iter_jsonl(args.input):
        stats["total_samples"] += 1

        sample_policy = sample.get("sample_translation_policy")
        stats["sample_translation_policy"][bool_bucket(sample_policy)] += 1
        stats["sample_translation_policy_reason"][clean_label(sample.get("sample_translation_policy_reason"))] += 1
        stats["sample_translation_policy_source"][clean_label(sample.get("sample_translation_policy_source"))] += 1

        for field in TEXT_FIELDS:
            value = sample.get(field)
            if not isinstance(value, str):
                stats["text_fields"][field]["missing_or_invalid"] += 1
            elif value.strip():
                stats["text_fields"][field]["not_blank"] += 1
            else:
                stats["text_fields"][field]["blank"] += 1

        input_is_blank = not isinstance(sample.get("input"), str) or not sample.get("input", "").strip()
        stats["input_field"]["blank" if input_is_blank else "not_blank"] += 1

        for field in SCALAR_LABEL_FIELDS:
            stats["labels"][field][clean_label(sample.get(field))] += 1
        for field in LIST_LABEL_FIELDS:
            for label in iter_clean_list(sample.get(field)):
                stats["labels"][field][label] += 1

        field_policy = sample.get("field_translation_policy")
        if not isinstance(field_policy, dict):
            stats["field_policy_shape"]["missing_or_invalid"] += 1
            for field in TEXT_FIELDS:
                stats["field_should_translate"][field]["missing"] += 1
            continue

        stats["field_policy_shape"]["present"] += 1
        sample_has_preserved_field = False
        sample_has_nonblank_preserved_field = False
        for field in TEXT_FIELDS:
            field_obj = field_policy.get(field)
            if not isinstance(field_obj, dict):
                stats["field_policy_shape"][f"{field}_missing_or_invalid"] += 1
                stats["field_should_translate"][field]["missing"] += 1
                continue

            all_languages = field_obj.get("all_languages")
            if not isinstance(all_languages, dict):
                stats["field_policy_shape"][f"{field}_all_languages_missing_or_invalid"] += 1
                all_languages = {}

            should_translate = all_languages.get("should_translate")
            stats["field_should_translate"][field][bool_bucket(should_translate)] += 1
            if should_translate is False:
                sample_has_preserved_field = True
                field_value = sample.get(field)
                if isinstance(field_value, str) and field_value.strip():
                    sample_has_nonblank_preserved_field = True
            if should_translate is False and field == "input" and not input_is_blank and sample_policy is True:
                stats["input_field"]["not_blank_and_sample_true_field_false"] += 1

            reason = all_languages.get("reason")
            if reason is not None:
                stats["field_should_translate_reason"][field][clean_label(reason)] += 1

            per_language = field_obj.get("per_language")
            if not isinstance(per_language, dict):
                stats["field_policy_shape"][f"{field}_per_language_missing_or_invalid"] += 1
                continue

            for language, decision in per_language.items():
                language_key = clean_label(language)
                if not isinstance(decision, dict):
                    stats["per_language_should_translate"][language_key]["invalid"] += 1
                    stats["per_language_should_translate_by_field"][(language_key, field)]["invalid"] += 1
                    continue
                bucket = bool_bucket(decision.get("should_translate"))
                stats["per_language_should_translate"][language_key][bucket] += 1
                stats["per_language_should_translate_by_field"][(language_key, field)][bucket] += 1

        if sample_has_preserved_field:
            stats["field_preservation"]["samples_with_at_least_one_field_false"] += 1
        if sample_has_nonblank_preserved_field:
            stats["field_preservation"]["samples_with_at_least_one_nonblank_field_false"] += 1

    total = stats["total_samples"]
    print(f"Input: {args.input}")
    print(f"Total samples: {total}")
    print()

    print_counter("Sample translation policy:", stats["sample_translation_policy"], total)
    print_counter("Sample translation policy reasons:", stats["sample_translation_policy_reason"], total, args.top)
    print_counter("Sample translation policy sources:", stats["sample_translation_policy_source"], total, args.top)

    for field in SCALAR_LABEL_FIELDS + LIST_LABEL_FIELDS:
        print_counter(f"{field} counts:", stats["labels"][field], total, args.top)

    for field in TEXT_FIELDS:
        print_counter(f"{field} text field:", stats["text_fields"][field], total)

    print_counter("Input field summary:", stats["input_field"], total)
    print_counter("Field preservation summary:", stats["field_preservation"], total)
    print_counter("Field policy shape:", stats["field_policy_shape"], total)

    for field in TEXT_FIELDS:
        print_counter(f"{field} all_languages.should_translate:", stats["field_should_translate"][field], total)
        print_counter(f"{field} all_languages.reason:", stats["field_should_translate_reason"][field], total, args.top)

    if stats["per_language_should_translate"]:
        for language in sorted(stats["per_language_should_translate"]):
            print_counter(
                f"{language} per_language.should_translate:",
                stats["per_language_should_translate"][language],
            )
        for language, field in sorted(stats["per_language_should_translate_by_field"]):
            print_counter(
                f"{language} {field} per_language.should_translate:",
                stats["per_language_should_translate_by_field"][(language, field)],
            )


if __name__ == "__main__":
    main()
