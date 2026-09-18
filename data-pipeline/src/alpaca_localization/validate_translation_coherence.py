#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, Iterable, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from .languages import LANG_NAME_BY_TAG

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def count_nonempty_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def extract_first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def query_gemma_server(
    prompt: str,
    endpoint: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        import requests
    except Exception as e:
        print(f"[warn] Gemma query unavailable (requests import failed): {e}")
        return ""

    try:
        resp = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You validate multilingual instruction-following samples with strict "
                            "language-specific coherence checks. "
                            "Return JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            print(f"[warn] Gemma server error {resp.status_code}: {resp.text[:300]}")
            return ""
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if "choices" in data and data["choices"] and "text" in data["choices"][0]:
                return str(data["choices"][0]["text"]).strip()
            return ""
    except Exception as e:
        print(f"[warn] Gemma request failed: {type(e).__name__}: {e}")
        return ""


def infer_language_tag(path: str) -> str:
    m = re.search(r"/alpaca_cleaned/([^/]+)/", path)
    if m:
        return m.group(1).strip().lower()

    parent = os.path.basename(os.path.dirname(path)).strip().lower()
    if parent:
        return parent
    return "unknown"


def build_validation_prompt(
    sample: Dict,
    language_tag: str,
    language_name: str,
) -> str:
    instruction = str(sample.get("instruction", ""))
    inp = str(sample.get("input", ""))
    output = str(sample.get("output", ""))
    primary_task = str(sample.get("primary_task", "unknown"))

    return (
        "Task:\n"
        f"Validate one sample in language: {language_name}.\n\n"
        "Input sample:\n"
        f"primary_task: {primary_task}\n"
        f"instruction: {instruction}\n"
        f"input: {inp}\n"
        f"output: {output}\n\n"
        "Coherence criteria (language-specific):\n"
        f"- Judge coherence from the actual text in {language_name}, not from the abstract task idea.\n"
        f"- Set is_coherent=false if wording in {language_name} makes the instruction/input/output semantically wrong, contradictory, or unclear.\n"
        f"- Set is_coherent=false if there is severe grammar/orthography/syntax degradation that changes or blocks meaning in {language_name}.\n"
        f"- Set is_coherent=false if mistranslation/literal calque causes wrong meaning in {language_name}.\n"
        f"- Set is_coherent=false if language mixing or wrong-language text breaks task understanding in {language_name}.\n"
        "- If errors are minor and meaning remains clear and logically aligned, is_coherent can be true.\n\n"
        "Language lock (mandatory):\n"
        f"- Final corrected instruction/input/output must be in {language_name}.\n"
        "- Exception: keep spans in another language only when the task explicitly requires them (for example, translation tasks or quoted source text that must remain as source).\n"
        "- Keep named entities, code, URLs, numbers, and formal symbols unchanged unless clearly malformed.\n"
        f"- Outside those required spans, avoid language mixing and normalize wording to {language_name}.\n\n"
        "Task and constraint preservation:\n"
        f"- Treat explicit instruction constraints as part of the task meaning, including format, number of items, casing, punctuation restrictions, required/forbidden words, exact phrases, and requested response language.\n"
        f"- Set is_coherent=false if the instruction/input/output is internally inconsistent with an explicit instruction constraint, or if the output clearly violates one.\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        '  "is_coherent": true,\n'
        '  "issues": "",\n'
        '  "changes": {}\n'
        "}\n\n"
        "Rules:\n"
        "- issues must be a very short reason only when not coherent.\n"
        "- If coherent, issues must be an empty string.\n"
        "- Always correct grammar, orthography, punctuation, and syntax when needed.\n"
        "- Put edits only in changes. changes must include ONLY modified fields among instruction/input/output as optional keys.\n"
        "- If nothing needs editing, return changes as {}.\n"
        "- If is_coherent=false, changes must repair the sample so it becomes coherent.\n"
        "- If is_coherent=true, changes can still contain language fixes.\n"
        "- Keep text in the dataset language, except required cross-language spans.\n"
        "- Preserve original intent, constraints, factual content, and task type whenever possible.\n"
        "- Do not add new requirements not implied by the sample.\n"
        "- Keep URLs, code, numbers, and named entities unless clearly malformed.\n"
        "- Keep empty input empty when appropriate.\n"
        "- Do not use markdown fences. Do not return multiple JSON objects. Do not explain.\n"
    )


def _normalize_bool(v: object, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"true", "1", "yes"}:
            return True
        if t in {"false", "0", "no"}:
            return False
    return default


def _normalize_issues(raw: object, is_coherent: bool) -> str:
    if is_coherent:
        return ""

    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts = [str(x).strip() for x in raw if str(x).strip()]
        text = "; ".join(parts)
    elif raw is None:
        text = ""
    else:
        text = str(raw)

    text = " ".join(text.strip().split())
    if not text:
        text = "no coherent task alignment"
    return text


def _sample_text(sample: Dict, field: str) -> str:
    value = sample.get(field, "")
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_output_sample_id(sample: Dict, line_no: int) -> object:
    if "sample_id" in sample:
        return sample.get("sample_id")
    if "id" in sample:
        return sample.get("id")
    if "instance_id" in sample:
        return sample.get("instance_id")
    return line_no


def _sample_id_key(sample_id: object) -> str:
    try:
        return json.dumps(sample_id, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(sample_id)


def _get_last_output_sample_id(path: str) -> Optional[object]:
    last_nonempty = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                last_nonempty = s

    if not last_nonempty:
        return None

    try:
        obj = json.loads(last_nonempty)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    if "sample_id" in obj:
        return obj.get("sample_id")
    if "id" in obj:
        return obj.get("id")
    if "instance_id" in obj:
        return obj.get("instance_id")
    return None


def _get_last_quarantine_sample_id(path: str) -> Optional[object]:
    last_nonempty = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                last_nonempty = s

    if not last_nonempty:
        return None

    try:
        obj = json.loads(last_nonempty)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    return obj.get("sample_id")


def _count_records_until_sample_id(input_path: str, target_sample_id: object) -> int:
    target_key = _sample_id_key(target_sample_id)
    seen = 0
    for line_no, sample in iter_jsonl(input_path):
        seen += 1
        sid = _resolve_output_sample_id(sample, line_no=line_no)
        if _sample_id_key(sid) == target_key:
            return seen
    return -1


def apply_validation_result(sample: Dict, parsed: Optional[dict], line_no: int) -> Tuple[Dict, str]:
    original_instruction = _sample_text(sample, "instruction")
    original_input = _sample_text(sample, "input")
    original_output = _sample_text(sample, "output")

    final_instruction = original_instruction
    final_input = original_input
    final_output = original_output
    parse_status = "ok"

    if not isinstance(parsed, dict):
        parse_status = "parse_failed"
        is_coherent = True
        issues = ""
    else:
        if "is_coherent" in parsed:
            is_coherent = _normalize_bool(parsed.get("is_coherent"), default=True)
        elif "flag_non_coherent" in parsed:
            non_coherent = _normalize_bool(parsed.get("flag_non_coherent"), default=False)
            is_coherent = not non_coherent
        else:
            is_coherent = True
        issues = _normalize_issues(parsed.get("issues"), is_coherent=is_coherent)

        model_changes = parsed.get("changes")
        if isinstance(model_changes, dict):
            model_instruction = model_changes.get("instruction")
            model_input = model_changes.get("input")
            model_output = model_changes.get("output")
        else:
            model_instruction = parsed.get("instruction")
            model_input = parsed.get("input")
            model_output = parsed.get("output")

        if isinstance(model_instruction, str):
            final_instruction = model_instruction
        if isinstance(model_input, str):
            final_input = model_input
        if isinstance(model_output, str):
            final_output = model_output

        # For coherent=true we expect all fields, but keep originals when omitted.
        # For coherent=false omitted fields are considered unchanged.

    instruction_changed = final_instruction != original_instruction
    input_changed = final_input != original_input
    output_changed = final_output != original_output

    post_edit: Dict[str, object] = {
        "instruction_changed": instruction_changed,
        "input_changed": input_changed,
        "output_changed": output_changed,
    }
    if instruction_changed:
        post_edit["instruction_original"] = original_instruction
    if input_changed:
        post_edit["input_original"] = original_input
    if output_changed:
        post_edit["output_original"] = original_output

    # Keep output in the same structure as input, but prepend new fields.
    enriched: Dict[str, object] = {
        "sample_id": _resolve_output_sample_id(sample=sample, line_no=line_no),
        "is_coherent": is_coherent,
        "issues": issues,
    }
    for k, v in sample.items():
        if k in {"sample_id", "is_coherent", "issues", "post_edit"}:
            continue
        if k == "instruction":
            enriched[k] = final_instruction
        elif k == "input":
            enriched[k] = final_input
        elif k == "output":
            enriched[k] = final_output
        else:
            enriched[k] = v
    enriched["post_edit"] = post_edit
    return enriched, parse_status


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-translation validation with Gemma: checks instruction/input/output coherence "
            "and applies grammar/orthography/syntax corrections."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--language-tag",
        default="",
        help="Language tag for the dataset. If empty, inferred from input path.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for quick tests.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress print interval.")
    parser.add_argument("--gemma-endpoint", default="http://127.0.0.1:10001/chat/completions")
    parser.add_argument("--gemma-timeout", type=int, default=130)
    parser.add_argument("--gemma-max-tokens", type=int, default=320)
    parser.add_argument("--gemma-temperature", type=float, default=0.0)
    parser.add_argument(
        "--error-raw-chars",
        type=int,
        default=4000,
        help="Max raw Gemma response chars stored in quarantine records. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--quarantine-filename",
        default="alpaca_post_correction_quarantine.jsonl",
        help="Quarantine filename inside <output-dir>/quarantine/.",
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    language_tag = (args.language_tag or infer_language_tag(args.input)).lower()
    language_name = LANG_NAME_BY_TAG.get(language_tag, language_tag)

    out_parent = os.path.dirname(os.path.abspath(args.output))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    quarantine_dir = os.path.join(out_parent, "quarantine")
    quarantine_path = os.path.join(quarantine_dir, args.quarantine_filename)

    resume_skip = 0
    if args.resume and os.path.exists(args.output):
        mode = "a"
        quarantine_mode = "a"
        resume_candidates = []
        last_output_sample_id = _get_last_output_sample_id(args.output)
        if last_output_sample_id is not None:
            by_id_skip = _count_records_until_sample_id(args.input, last_output_sample_id)
            if by_id_skip >= 0:
                resume_candidates.append(("output", last_output_sample_id, by_id_skip))
            else:
                print(
                    "[warn] Resume output sample_id not found in input; "
                    "falling back to line-count resume if no quarantine candidate is available."
                )

        if os.path.exists(quarantine_path):
            last_quarantine_sample_id = _get_last_quarantine_sample_id(quarantine_path)
            if last_quarantine_sample_id is not None:
                by_id_skip = _count_records_until_sample_id(args.input, last_quarantine_sample_id)
                if by_id_skip >= 0:
                    resume_candidates.append(("quarantine", last_quarantine_sample_id, by_id_skip))
                else:
                    print("[warn] Resume quarantine sample_id not found in input.")

        if resume_candidates:
            source, sample_id, resume_skip = max(resume_candidates, key=lambda x: x[2])
            print(
                "[info] Resume enabled (by sample_id). "
                f"Last processed sample_id from {source}: {sample_id!r}. "
                f"Skipping first {resume_skip} input records."
            )
        else:
            resume_skip = count_nonempty_lines(args.output)
            print(
                "[warn] Could not read last sample_id from output or quarantine; "
                f"falling back to line-count resume ({resume_skip} records)."
            )
    else:
        mode = "w"
        quarantine_mode = "w"

    total_records = count_nonempty_lines(args.input)
    resume_skip = min(resume_skip, total_records)
    total_to_process = max(0, total_records - resume_skip)
    if args.max_records > 0:
        total_to_process = min(total_to_process, args.max_records)

    pbar = None
    if not args.disable_tqdm and tqdm is not None:
        pbar = tqdm(total=total_to_process, desc="validate", unit="sample", dynamic_ncols=True)
    elif not args.disable_tqdm and tqdm is None:
        print("[warn] tqdm is not installed; progress bar disabled.")

    print(f"[info] Input: {args.input}")
    print(f"[info] Output: {args.output}")
    print(f"[info] Quarantine: {quarantine_path}")
    print(f"[info] Language: {language_tag} ({language_name})")
    print(f"[info] Gemma endpoint: {args.gemma_endpoint}")

    processed = 0
    parse_failed = 0

    os.makedirs(quarantine_dir, exist_ok=True)
    with open(args.output, mode, encoding="utf-8") as out_f, open(
        quarantine_path, quarantine_mode, encoding="utf-8"
    ) as quarantine_f:
        for idx, (_line_no, sample) in enumerate(iter_jsonl(args.input), start=1):
            if idx <= resume_skip:
                continue
            if args.max_records > 0 and processed >= args.max_records:
                break

            prompt = build_validation_prompt(
                sample=sample,
                language_tag=language_tag,
                language_name=language_name,
            )
            raw = query_gemma_server(
                prompt=prompt,
                endpoint=args.gemma_endpoint,
                timeout=args.gemma_timeout,
                max_tokens=args.gemma_max_tokens,
                temperature=args.gemma_temperature,
            )
            parsed = extract_first_json_object(raw)
            enriched, status = apply_validation_result(sample=sample, parsed=parsed, line_no=_line_no)
            if status != "ok":
                parse_failed += 1
                raw_preview = truncate_text(raw, args.error_raw_chars)
                quarantine_record = {
                    "line_no": _line_no,
                    "sample_id": _resolve_output_sample_id(sample=sample, line_no=_line_no),
                    "language": language_tag,
                    "error": "Gemma response is not a valid JSON object",
                    "raw_gemma_response": raw_preview,
                    "source_sample": sample,
                    "output_sample": enriched,
                }
                quarantine_f.write(json.dumps(quarantine_record, ensure_ascii=False) + "\n")
                quarantine_f.flush()
                print(
                    f"[warn] line={_line_no} lang={language_tag} quarantined after failed parse. "
                    f"quarantine={quarantine_path}"
                )
                processed += 1
                if pbar is not None:
                    pbar.update(1)
                if processed % args.log_every == 0:
                    print(
                        f"[info] Processed {processed} samples... "
                        f"(parse_failed={parse_failed})"
                    )
                continue
            out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            processed += 1
            if pbar is not None:
                pbar.update(1)
            if processed % args.log_every == 0:
                print(
                    f"[info] Processed {processed} samples... "
                    f"(parse_failed={parse_failed})"
                )

    if pbar is not None:
        pbar.close()

    print(
        f"[done] Wrote {processed} records to: {args.output} "
        f"(parse_failed={parse_failed})"
    )


if __name__ == "__main__":
    main()
