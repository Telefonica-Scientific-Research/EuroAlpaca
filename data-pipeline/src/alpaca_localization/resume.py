"""Safety checks for resuming line-oriented JSONL pipeline stages."""

import json
from pathlib import Path
from typing import Dict, Iterator, Tuple


def _iter_jsonl_records(path: str) -> Iterator[Tuple[int, Dict]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in resume file {path} at line {line_no}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Expected a JSON object in resume file {path} at line {line_no}."
                )
            yield line_no, record


def _sample_id(record: Dict) -> str | None:
    if "sample_id" not in record:
        return None
    return json.dumps(record["sample_id"], ensure_ascii=False, sort_keys=True)


def validate_resume_prefix(input_path: str, output_path: str) -> int:
    """Validate that existing output is a sample-ID-compatible input prefix.

    Returns the number of completed records. Both stages support records without
    sample IDs for direct use, but when either side has an ID, both must have the
    same ID at the same position.
    """
    input_records = _iter_jsonl_records(input_path)
    completed = 0

    for output_line, output_record in _iter_jsonl_records(output_path):
        try:
            input_line, input_record = next(input_records)
        except StopIteration as error:
            raise RuntimeError(
                f"Cannot safely resume {output_path}: existing output has more "
                f"records than input {input_path}. Use a new empty ALPACA_ROOT "
                "or rerun with RESUME=false."
            ) from error

        input_id = _sample_id(input_record)
        output_id = _sample_id(output_record)
        if input_id != output_id:
            raise RuntimeError(
                f"Cannot safely resume {output_path}: sample_id mismatch at "
                f"output line {output_line} and input line {input_line} "
                f"(output={output_id!r}, input={input_id!r}). Use a new empty "
                "ALPACA_ROOT or rerun with RESUME=false."
            )
        completed += 1

    return completed
