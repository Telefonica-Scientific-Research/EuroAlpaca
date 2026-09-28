#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            obj = json.loads(row)
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_no}, got {type(obj)}")
            yield line_no, obj


def resolve_sample_id(sample: Dict, fallback_line_no: int) -> object:
    if "sample_id" in sample:
        return sample["sample_id"]
    if "id" in sample:
        return sample["id"]
    if "instance_id" in sample:
        return sample["instance_id"]
    return fallback_line_no


def load_index(path: str, label: str) -> Dict[object, Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label} file: {path}")
    out: Dict[object, Dict] = {}
    duplicates = 0
    for line_no, obj in iter_jsonl(path):
        sid = resolve_sample_id(obj, fallback_line_no=line_no)
        if sid in out:
            duplicates += 1
        out[sid] = obj
    if duplicates > 0:
        raise RuntimeError(f"{label} has duplicate sample_id values: {duplicates}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble per-language pre-validation dataset by merging stage-4 translated-kept "
            "samples with stage-5 localized-discarded samples."
        )
    )
    parser.add_argument("--translated", required=True, help="Stage-4 per-language JSONL path.")
    parser.add_argument("--localized-discarded", required=True, help="Stage-5 per-language JSONL path.")
    parser.add_argument(
        "--expected-source",
        required=True,
        help="Stage-3 annotated source JSONL used to verify the complete sample_id set.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (normally alpaca_cleaned_labeled_annotated_gemma4.jsonl).",
    )
    parser.add_argument("--language", required=True, help="Target language tag for output rows.")
    args = parser.parse_args()

    translated_by_id = load_index(args.translated, "translated")
    localized_by_id = load_index(args.localized_discarded, "localized_discarded")
    expected_by_id = load_index(args.expected_source, "expected_source")

    overlap = set(translated_by_id).intersection(localized_by_id)
    if overlap:
        example = sorted(list(overlap), key=lambda x: str(x))[:10]
        raise RuntimeError(
            f"sample_id overlap between translated and localized_discarded: "
            f"{len(overlap)} ids, examples={example}"
        )

    assembled_ids = set(translated_by_id).union(localized_by_id)
    expected_ids = set(expected_by_id)
    missing = expected_ids.difference(assembled_ids)
    unexpected = assembled_ids.difference(expected_ids)
    if missing or unexpected:
        missing_examples = sorted(missing, key=lambda x: str(x))[:10]
        unexpected_examples = sorted(unexpected, key=lambda x: str(x))[:10]
        raise RuntimeError(
            "Stage-6 sample_id completeness check failed: "
            f"expected={len(expected_ids)}, assembled={len(assembled_ids)}, "
            f"missing={len(missing)} examples={missing_examples}, "
            f"unexpected={len(unexpected)} examples={unexpected_examples}. "
            "Resolve stage-4/stage-5 failures or quarantined samples before assembly."
        )

    # The stage-3 source defines the canonical dataset order. Iterating it also
    # avoids lexicographical ID sorting such as 1, 10, 100, 2.
    merged_rows: List[Dict] = []
    for sid in expected_by_id:
        if sid in translated_by_id:
            obj = translated_by_id[sid]
        else:
            obj = localized_by_id[sid]
        row = dict(obj)
        row["sample_id"] = sid
        row["language"] = args.language
        merged_rows.append(row)

    out_parent = os.path.dirname(os.path.abspath(args.output))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for row in merged_rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[done] assembled language={args.language} translated={len(translated_by_id)} "
        f"localized_discarded={len(localized_by_id)} total={len(merged_rows)} output={args.output}"
    )


if __name__ == "__main__":
    main()
