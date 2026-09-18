#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, Iterable, Optional, Tuple


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object at line {line_no}, got {type(obj)}")
            yield line_no, obj


def infer_language_tag(path: str) -> str:
    m = re.search(r"/alpaca_cleaned/([^/]+)/", path)
    if m:
        return m.group(1).strip().lower()
    parent = os.path.basename(os.path.dirname(path)).strip().lower()
    if parent:
        return parent
    return "unknown"


def resolve_sample_id(sample: Dict, fallback_line_no: int) -> object:
    if "sample_id" in sample:
        return sample.get("sample_id")
    if "id" in sample:
        return sample.get("id")
    if "instance_id" in sample:
        return sample.get("instance_id")
    return fallback_line_no


def to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def pick_field_value(field: str, original: Dict, validated: Optional[Dict]) -> str:
    original_text = to_text(original.get(field, ""))
    if not isinstance(validated, dict):
        return original_text

    post_edit = validated.get("post_edit")
    if isinstance(post_edit, dict):
        changed = post_edit.get(f"{field}_changed")
        if changed is False:
            return original_text
        if changed is True:
            v = validated.get(field)
            if isinstance(v, str):
                return v
            return original_text

    # Fallback when post_edit metadata is unavailable.
    v = validated.get(field)
    if isinstance(v, str) and v != original_text:
        return v
    return original_text


def pick_language(original: Dict, validated: Optional[Dict], fallback_lang: str) -> str:
    orig_lang = original.get("language")
    if isinstance(orig_lang, str) and orig_lang.strip():
        return orig_lang
    if isinstance(validated, dict):
        val_lang = validated.get("language")
        if isinstance(val_lang, str) and val_lang.strip():
            return val_lang
    return fallback_lang


def build_validated_index(validated_path: str) -> Dict[object, Dict]:
    by_id: Dict[object, Dict] = {}
    for line_no, sample in iter_jsonl(validated_path):
        sid = resolve_sample_id(sample, fallback_line_no=line_no)
        if sid in by_id:
            raise RuntimeError(
                f"Validated file has duplicate sample_id {sid!r} at line {line_no}."
            )
        by_id[sid] = sample
    return by_id


def collect_original_ids(original_path: str) -> set:
    sample_ids = set()
    for line_no, sample in iter_jsonl(original_path):
        sid = resolve_sample_id(sample, fallback_line_no=line_no)
        if sid in sample_ids:
            raise RuntimeError(
                f"Original file has duplicate sample_id {sid!r} at line {line_no}."
            )
        sample_ids.add(sid)
    return sample_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge original annotated JSONL and post-validated JSONL, keeping only "
            "sample_id/instruction/input/output/language."
        )
    )
    parser.add_argument("--original", required=True, help="Original annotated JSONL path.")
    parser.add_argument("--validated", required=True, help="Post-validated JSONL path.")
    parser.add_argument("--output", required=True, help="Output merged JSONL path.")
    parser.add_argument(
        "--language",
        default="",
        help="Fallback language value. If empty, inferred from --original path.",
    )
    parser.add_argument("--log-every", type=int, default=10000)
    args = parser.parse_args()

    if not os.path.exists(args.original):
        raise FileNotFoundError(f"Original file does not exist: {args.original}")
    if not os.path.exists(args.validated):
        raise FileNotFoundError(f"Validated file does not exist: {args.validated}")

    fallback_lang = (args.language or infer_language_tag(args.original)).strip().lower()
    out_parent = os.path.dirname(os.path.abspath(args.output))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    validated_by_id = build_validated_index(args.validated)
    original_ids = collect_original_ids(args.original)
    unexpected_ids = set(validated_by_id).difference(original_ids)
    if unexpected_ids:
        examples = sorted(unexpected_ids, key=lambda value: str(value))[:10]
        raise RuntimeError(
            f"Validated file contains {len(unexpected_ids)} sample_id values absent "
            f"from the original file, examples={examples}."
        )

    processed = 0
    matched = 0
    changed_samples = 0
    changed_instruction = 0
    changed_input = 0
    changed_output = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for line_no, orig in iter_jsonl(args.original):
            sid = resolve_sample_id(orig, fallback_line_no=line_no)
            val = validated_by_id.get(sid)
            if val is not None:
                matched += 1

            final_instruction = pick_field_value("instruction", orig, val)
            final_input = pick_field_value("input", orig, val)
            final_output = pick_field_value("output", orig, val)
            final_language = pick_language(orig, val, fallback_lang=fallback_lang)

            orig_instruction = to_text(orig.get("instruction", ""))
            orig_input = to_text(orig.get("input", ""))
            orig_output = to_text(orig.get("output", ""))

            instruction_changed = final_instruction != orig_instruction
            input_changed = final_input != orig_input
            output_changed = final_output != orig_output

            if instruction_changed:
                changed_instruction += 1
            if input_changed:
                changed_input += 1
            if output_changed:
                changed_output += 1
            if instruction_changed or input_changed or output_changed:
                changed_samples += 1

            merged = {
                "sample_id": sid,
                "instruction": final_instruction,
                "input": final_input,
                "output": final_output,
                "language": final_language,
            }
            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")

            processed += 1
            if processed % args.log_every == 0:
                print(
                    f"[info] Processed {processed} samples... "
                    f"(matched_validated={matched}, changed_samples={changed_samples})"
                )

    print(f"[done] Wrote {processed} merged samples to: {args.output}")
    print(f"[done] Matched validated samples by sample_id: {matched}")
    print(
        "[done] Changed fields: "
        f"instruction={changed_instruction}, input={changed_input}, output={changed_output}, "
        f"samples_with_any_change={changed_samples}"
    )


if __name__ == "__main__":
    main()
