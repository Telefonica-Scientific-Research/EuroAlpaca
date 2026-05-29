#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, TextIO

ALL_DOMAIN_TYPES: List[str] = [
    "general",
    "programming",
    "technology",
    "science",
    "mathematics",
    "medicine",
    "law",
    "business",
    "education",
    "humanities",
    "social_science",
    "creative",
    "other",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {exc}") from exc


def open_domain_writers(output_dir: Path) -> Dict[str, TextIO]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: Dict[str, TextIO] = {}
    for domain in ALL_DOMAIN_TYPES:
        out_path = output_dir / f"{domain}.jsonl"
        writers[domain] = out_path.open("w", encoding="utf-8")
    return writers


def close_domain_writers(writers: Dict[str, TextIO]) -> None:
    for handle in writers.values():
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a labeled Alpaca JSONL into one JSONL file per primary_domain."
    )
    parser.add_argument(
        "--input",
        default="alpaca_cleaned_labeled.jsonl",
        help="Path to labeled JSONL file containing a primary_domain field.",
    )
    parser.add_argument(
        "--output-dir",
        default="domains",
        help="Directory where domain-specific JSONL files will be written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Open one file handle per task type to avoid reopening files repeatedly
    writers = open_domain_writers(output_dir)
    counts = {domain: 0 for domain in ALL_DOMAIN_TYPES}
    unknown_count = 0

    try:
        for line_no, sample in iter_jsonl(input_path):
            primary_domain = sample.get("primary_domain")
            if not isinstance(primary_domain, str):
                raise RuntimeError(f"Line {line_no}: missing or invalid primary_domain.")

            if primary_domain not in writers:
                primary_domain = "other"
                unknown_count += 1

            writers[primary_domain].write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[primary_domain] += 1
    finally:
        close_domain_writers(writers)

    print(f"[done] Wrote domain files to: {output_dir.resolve()}")
    for domain in ALL_DOMAIN_TYPES:
        print(f"{domain}: {counts[domain]}")
    if unknown_count:
        print(
            f"[warn] Found {unknown_count} samples with unknown primary_domain; wrote them to other.jsonl"
        )


if __name__ == "__main__":
    main()
