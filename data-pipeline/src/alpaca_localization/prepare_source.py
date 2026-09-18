#!/usr/bin/env python3
"""Normalize Alpaca Cleaned JSON or JSONL into pipeline-ready JSONL."""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


REQUIRED_TEXT_FIELDS = ("instruction", "input", "output")


def read_records(path: Path) -> Iterable[dict]:
    """Read either a JSON array or one-JSON-object-per-line input file."""
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            character = handle.read(1)
            if not character or not character.isspace():
                first = character
                break
        handle.seek(0)
        if first == "[":
            records = json.load(handle)
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for index, record in enumerate(records, 1):
                if not isinstance(record, dict):
                    raise ValueError(f"{path}: record {index} is not a JSON object")
                yield record
            return

        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


def normalize_record(record: dict, index: int) -> dict:
    """Validate the Alpaca fields and add a stable sample identifier."""
    normalized = dict(record)
    for field in REQUIRED_TEXT_FIELDS:
        value = normalized.get(field, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(
                f"Record {index}: {field!r} must be a string, got "
                f"{type(value).__name__}"
            )
        normalized[field] = value

    if "sample_id" not in normalized:
        normalized["sample_id"] = index
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Alpaca Cleaned JSON or JSONL to pipeline-ready JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional record limit for a quick pipeline smoke test.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Source dataset not found: {args.input}")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    seen_ids = set()
    written = 0
    try:
        with temporary_output.open("w", encoding="utf-8") as output_handle:
            for index, record in enumerate(read_records(args.input), 1):
                normalized = normalize_record(record, index)
                sample_id = json.dumps(
                    normalized["sample_id"], ensure_ascii=False, sort_keys=True
                )
                if sample_id in seen_ids:
                    raise ValueError(
                        f"Record {index}: duplicate sample_id {normalized['sample_id']!r}"
                    )
                seen_ids.add(sample_id)
                output_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                written += 1
                if args.limit and written >= args.limit:
                    break
        if written == 0:
            raise ValueError(f"Source dataset contains no records: {args.input}")
        os.replace(temporary_output, args.output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    print(f"[done] Wrote {written} records to: {args.output}")


if __name__ == "__main__":
    main()
