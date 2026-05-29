#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from typing import List, Literal, Optional, Tuple, Dict, Any

import requests

# -----------------------------
# 1) Taxonomies
# -----------------------------
TaskType = Literal[
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

DomainType = Literal[
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

# -----------------------------
# 2) Prompt builder
# -----------------------------
ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:
"""

CLASSIFIER_INSTRUCTIONS = """You are labeling BOTH the TASK TYPE(S) and the DOMAIN(S) of the request.

You must pick:
- one PRIMARY task type (exactly one)
- zero or more SECONDARY task types (may be empty)
- one PRIMARY domain (exactly one)
- zero or more SECONDARY domains (may be empty)

Allowed task types (use EXACT strings):
{task_types}

Allowed domain types (use EXACT strings):
{domain_types}

Rules (tasks):
- primary_task must be exactly one of the allowed task types.
- secondary_tasks must be a JSON array of allowed task types.
- secondary_tasks must NOT repeat primary_task.
- If uncertain, use "other".

Rules (domains):
- primary_domain must be exactly one of the allowed domain types.
- secondary_domains must be a JSON array of allowed domain types.
- secondary_domains must NOT repeat primary_domain.
- Use "general" when the topic is everyday / non-specialized.
- Use "other" only if none fit.

Return ONLY valid JSON with exactly these keys:
{{
  "primary_task": "<one_task_type>",
  "secondary_tasks": ["<task_type>", "..."],
  "primary_domain": "<one_domain_type>",
  "secondary_domains": ["<domain_type>", "..."]
}}

No extra text. No markdown. No code fences.
"""

def build_prompt(instruction: str, inp: str) -> str:
    alpaca = ALPACA_TEMPLATE.format(instruction=instruction.strip(), input=(inp or "").strip())
    taxonomy = CLASSIFIER_INSTRUCTIONS.format(
        task_types=", ".join(ALL_TASK_TYPES),
        domain_types=", ".join(ALL_DOMAIN_TYPES),
    )
    return taxonomy + "\n\n---\n\n" + alpaca

# -----------------------------
# 3) Local Gemma server query
# -----------------------------
def query_gemma_server(
    prompt: str,
    max_tokens: int = 160,          # bumped slightly since we now output 4 fields
    temperature: float = 0.0,
    stream: bool = False,
    endpoint: str = "http://127.0.0.1:10006/chat/completions",
    timeout: int = 130,
) -> str:
    """
    Query a local OpenAI-compatible /chat/completions endpoint (Gemma server on port 10006).
    """
    try:
        resp = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json={
                "messages": [
                    {"role": "system", "content": "You are a careful dataset annotator. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            },
            timeout=timeout,
        )

        if resp.status_code != 200:
            print(f"[warn] Server error {resp.status_code}: {resp.text[:500]}")
            return ""

        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if "choices" in data and data["choices"] and "text" in data["choices"][0]:
                return str(data["choices"][0]["text"]).strip()
            return ""

    except requests.exceptions.Timeout:
        print("[warn] Timeout querying Gemma server")
        return ""
    except Exception as e:
        print(f"[warn] Exception querying Gemma server: {type(e).__name__}: {e}")
        return ""

# -----------------------------
# 4) JSON extraction + validation
# -----------------------------
# It looks for text starting with { and ending with }
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json_object(text: str) -> Optional[dict]:
    """
    Tries to recover JSON if the model accidentally adds extra text.
    We look for the first {...} block and parse it.
    """
    if not text:
        return None

    text = text.strip()

    # Fast path: already pure JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Recovery path: grab first {...}
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None

    # Gets the matched substring
    candidate = m.group(0).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None

def _clean_list_of_labels(
    labels: Any,
    allowed: List[str],
    primary_value: str,
) -> List[str]:
    if not isinstance(labels, list):
        return []
    cleaned: List[str] = []
    for t in labels:
        if isinstance(t, str):
            t = t.strip()
            if t in allowed and t != primary_value and t not in cleaned:
                cleaned.append(t)
    return cleaned

def normalize_and_validate_all(obj: dict) -> Tuple[str, List[str], str, List[str]]:
    """
    Returns (primary_task, secondary_tasks, primary_domain, secondary_domains)
    with strict validation and safe fallbacks.
    """
    # --- tasks ---
    primary_task = obj.get("primary_task", "other") # fallback to "other" if missing or invalid
    secondary_tasks = obj.get("secondary_tasks", []) # fallback to empty list if missing or invalid

    if not isinstance(primary_task, str):
        primary_task = "other" # fallback to "other" if not a string
    primary_task = primary_task.strip()
    if primary_task not in ALL_TASK_TYPES:
        primary_task = "other" # fallback to "other" if invalid

    cleaned_secondary_tasks = _clean_list_of_labels(
        secondary_tasks, ALL_TASK_TYPES, primary_task
    )

    # --- domains ---
    primary_domain = obj.get("primary_domain", "general") # fallback to "general" if missing or invalid
    secondary_domains = obj.get("secondary_domains", []) # fallback to empty list if missing or invalid

    if not isinstance(primary_domain, str):
        primary_domain = "general" # fallback to "general" if not a string
    primary_domain = primary_domain.strip()
    if primary_domain not in ALL_DOMAIN_TYPES:
        primary_domain = "general" # fallback to "general" if invalid

    cleaned_secondary_domains = _clean_list_of_labels(
        secondary_domains, ALL_DOMAIN_TYPES, primary_domain
    )

    return primary_task, cleaned_secondary_tasks, primary_domain, cleaned_secondary_domains

# -----------------------------
# 5) Robust retry/backoff
# -----------------------------
def call_with_backoff(fn, max_retries: int = 8, base_sleep: float = 0.8):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            sleep = base_sleep * (2 ** attempt) + random.uniform(0, 0.25)
            print(f"[warn] Call failed ({type(e).__name__}): {e}. Retrying in {sleep:.2f}s...")
            time.sleep(sleep)

# -----------------------------
# 6) JSONL helpers
# -----------------------------
# Generator that reads a JSONL file
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}") from e

# Count non-empty lines in a file (for resume functionality)
def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

# -----------------------------
# 7) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Label Alpaca JSONL with task + domain types using a local Gemma server.")
    parser.add_argument("--input", required=True, help="Input JSONL (instruction,input,output).")
    parser.add_argument("--output", required=True, help="Output JSONL (adds primary/secondary task + domain fields).")
    parser.add_argument("--endpoint", default="http://127.0.0.1:10006/chat/completions", help="Chat completions endpoint.")
    parser.add_argument("--max-output-tokens", type=int, default=160, help="Max tokens for classification output.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (0 recommended for labeling).")
    parser.add_argument("--resume", action="store_true", help="Resume by skipping already-written output lines.")
    parser.add_argument("--timeout", type=int, default=130, help="HTTP timeout seconds.")
    args = parser.parse_args()

    already_done = 0
    if args.resume and os.path.exists(args.output):
        already_done = count_lines(args.output)
        print(f"[info] Resuming: output already has {already_done} lines; will skip those inputs.")

    with open(args.output, "a" if args.resume else "w", encoding="utf-8") as out_f:
        for idx, (line_no, sample) in enumerate(iter_jsonl(args.input), start=1):
            if idx <= already_done:
                continue

            instruction = sample.get("instruction", "")
            inp = sample.get("input", "")
            if not isinstance(instruction, str) or not isinstance(inp, str):
                raise RuntimeError(f"Line {line_no}: 'instruction' and 'input' must be strings.")

            prompt = build_prompt(instruction, inp)

            def do_call_and_parse():
                raw = query_gemma_server(
                    prompt,
                    max_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    stream=False,
                    endpoint=args.endpoint,
                    timeout=args.timeout,
                )
                obj = extract_first_json_object(raw)
                if obj is None:
                    raise ValueError(f"Could not parse JSON from model output: {raw[:250]}")
                return obj

            obj = call_with_backoff(do_call_and_parse)
            p_task, s_tasks, p_dom, s_doms = normalize_and_validate_all(obj)

            sample["primary_task"] = p_task
            sample["secondary_tasks"] = s_tasks
            sample["primary_domain"] = p_dom
            sample["secondary_domains"] = s_doms

            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if idx % 50 == 0:
                print(f"[info] Processed {idx} examples...")

    print("[done] Wrote labeled JSONL to:", args.output)

if __name__ == "__main__":
    main()