#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, TextIO

ALL_TASK_TYPES: List[str] = [
    "question_answering",
    "reasoning",
    "summarization",
    "translation",
    "paraphrasing",
    "classification",
    "information_extraction",
    "creative_writing",
    "procedural_instructions",
    "coding",
    "style_transfer",
    "evaluation",
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


def open_task_writers(output_dir: Path) -> Dict[str, TextIO]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: Dict[str, TextIO] = {}
    for task in ALL_TASK_TYPES:
        out_path = output_dir / f"{task}.jsonl"
        writers[task] = out_path.open("w", encoding="utf-8")
    return writers


def close_task_writers(writers: Dict[str, TextIO]) -> None:
    for handle in writers.values():
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a labeled Alpaca JSONL into one JSONL file per primary_task."
    )
    parser.add_argument(
        "--input",
        default="alpaca_cleaned_labeled.jsonl",
        help="Path to labeled JSONL file containing a primary_task field.",
    )
    parser.add_argument(
        "--output-dir",
        default="tasks",
        help="Directory where task-specific JSONL files will be written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Open one file handle per task type to avoid reopening files repeatedly
    writers = open_task_writers(output_dir)
    counts = {task: 0 for task in ALL_TASK_TYPES}
    unknown_count = 0

    try:
        for line_no, sample in iter_jsonl(input_path):
            primary_task = sample.get("primary_task")
            if not isinstance(primary_task, str):
                raise RuntimeError(f"Line {line_no}: missing or invalid primary_task.")

            if primary_task not in writers:
                primary_task = "other"
                unknown_count += 1

            writers[primary_task].write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[primary_task] += 1
    finally:
        close_task_writers(writers)

    print(f"[done] Wrote task files to: {output_dir.resolve()}")
    for task in ALL_TASK_TYPES:
        print(f"{task}: {counts[task]}")
    if unknown_count:
        print(
            f"[warn] Found {unknown_count} samples with unknown primary_task; wrote them to other.jsonl"
        )


if __name__ == "__main__":
    main()
