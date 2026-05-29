"""JSONL, fallback, and parse-failure IO helpers."""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Iterable, Tuple

def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict]]:
    """Iterate over jsonl."""
    # Input contract: one JSON object per non-empty line.
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object at line {line_no}, got {type(obj)}")
            yield line_no, obj


def count_nonempty_lines(path: str) -> int:
    """Count nonempty lines."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def parse_failure_dir(args: argparse.Namespace) -> str:
    """Parse failure dir."""
    configured = getattr(args, "parse_failure_dir", "")
    if configured:
        return configured
    return f"{args.output}.parse_failures"


def quarantine_parse_failure(
    sample: Dict,
    args: argparse.Namespace,
    stage: str,
    prompt: str,
    raw_response: str,
) -> None:
    """Save per-sample Gemma JSON parse failures for later inspection."""
    out_dir = parse_failure_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    sample_id = str(sample.get("sample_id", "unknown"))
    path = os.path.join(out_dir, f"sample_{sample_id}_{stage}.json")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "sample_id": sample.get("sample_id"),
        "sample": sample,
        "prompt": prompt,
        "raw_response": raw_response,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

__all__ = [
    "iter_jsonl",
    "count_nonempty_lines",
    "parse_failure_dir",
    "quarantine_parse_failure",
]
