#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

LANG_NAME_BY_TAG = {
    "ast": "Asturian",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "co": "Corsican",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "fur": "Friulian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "gv": "Manx",
    "hr": "Croatian",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "ka": "Georgian",
    "lb": "Luxembourgish",
    "lij": "Ligurian",
    "lmo": "Lombard",
    "lt": "Lithuanian",
    "ltg": "Latgalian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "mt": "Maltese",
    "nl": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokmal",
    "oci": "Occitan",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sc": "Sardinian",
    "scn": "Sicilian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "szl": "Silesian",
    "uk": "Ukrainian",
    "vec": "Venetian",
}


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


def load_resume_state(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        line_no = data.get("last_input_line", 0)
        return int(line_no) if isinstance(line_no, int) or str(line_no).isdigit() else 0
    except Exception:
        return 0


def save_resume_state(path: str, last_input_line: int) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"last_input_line": last_input_line}, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def extract_first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()

    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for start in [idx for idx, char in enumerate(text) if char == "{"]:
        try:
            obj, _end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


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
                            "You localize instruction-following dataset samples. "
                            "Return one valid JSON object only. "
                            "Do not use markdown code fences."
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
            print(f"[warn] Gemma server error {resp.status_code}: {resp.text[:600]}")
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


def build_localization_prompt(sample: Dict, target_lang: str, target_lang_name: str) -> str:
    instruction = to_text(sample.get("instruction", ""))
    inp = to_text(sample.get("input", ""))
    output = to_text(sample.get("output", ""))
    primary_task = to_text(sample.get("primary_task", "unknown"))
    primary_domain = to_text(sample.get("primary_domain", "unknown"))
    secondary_tasks = sample.get("secondary_tasks", [])
    secondary_domains = sample.get("secondary_domains", [])

    return (
        "You are creating a localized counterpart of one English Alpaca-style sample that cannot be directly translated.\n"
        f"Target language: {target_lang_name} ({target_lang})\n\n"
        "Original sample metadata:\n"
        f"primary_task: {primary_task}\n"
        f"primary_domain: {primary_domain}\n"
        f"secondary_tasks: {secondary_tasks}\n"
        f"secondary_domains: {secondary_domains}\n\n"
        "Original English sample:\n"
        f"instruction: {instruction}\n"
        f"input: {inp}\n"
        f"output: {output}\n\n"
        "Localization rules:\n"
        f"- Write the final instruction, input, and output in {target_lang_name}.\n"
        "- Make the localized sample as semantically close to the original as the target language allows.\n"
        "- Treat the result as a near-parallel sample: preserve the same meaning, intent, task type, difficulty, constraints, and expected answer style whenever possible.\n"
        "- The first priority is that the sample makes sense in the target language and that the task logic is correct.\n"
        "- The second priority is semantic closeness to the original so the pair can reasonably be treated as parallel data.\n"
        "- Do not do literal machine translation when it would break the task logic or sound unnatural in the target language.\n"
        "- Adapt grammar, orthography, syntax, vocabulary, morphology, punctuation, and idioms to the target language.\n"
        "- For spelling, grammar, tense, voice, punctuation, typo, word-count, rhyme, or wordplay tasks, create a target-language equivalent with target-language examples and a correct target-language answer.\n"
        "- If the original sample uses an English-specific sentence, word, alphabet, proverb, idiom, or error, replace it with a natural equivalent in the target language.\n"
        "- When replacing English-specific material, keep the replacement as close as possible in topic, tone, length, complexity, and answer structure.\n"
        "- Preserve all functional information from the instruction/input fields, including separate lines, criteria, labels, options, constraints, and examples\n"
        "- Keep code, URLs, numbers, formal symbols, and named entities unchanged unless they are part of the language-specific exercise.\n"
        "- Keep empty input empty when the localized task does not need an input field.\n"
        "- The localized output must correctly answer the localized instruction/input, not the English source.\n"
        "- Do not mention that the sample was translated or localized.\n"
        "- Do not add explanations outside the requested output field.\n\n"
        "Return ONLY valid JSON with exactly this schema:\n"
        "{\n"
        '  "instruction": "...",\n'
        '  "input": "...",\n'
        '  "output": "..."\n'
        "}\n"
        "No markdown. No extra keys. Do not wrap JSON in code fences. "
        "Escape internal double quotes and newline characters correctly."
    )


def build_retry_prompt(
    original_prompt: str,
    previous_response: str,
    error: str,
) -> str:
    previous_response = previous_response.strip()
    if len(previous_response) > 2000:
        previous_response = previous_response[:2000] + "\n...[truncated]"
    return (
        f"{original_prompt}\n\n"
        "Your previous response could not be used.\n"
        f"Validation error: {error}\n\n"
        "Previous response:\n"
        f"{previous_response}\n\n"
        "Retry now. Return ONLY valid JSON with exactly these string fields:\n"
        "{\n"
        '  "instruction": "...",\n'
        '  "input": "...",\n'
        '  "output": "..."\n'
        "}\n"
        "No markdown. No comments. No extra keys. Do not wrap JSON in code fences."
    )


def validate_localized_fields(parsed: Optional[dict]) -> Tuple[Optional[Tuple[str, str, str]], str]:
    if not isinstance(parsed, dict):
        return None, "response is not a valid JSON object"

    missing = [field for field in ("instruction", "input", "output") if field not in parsed]
    if missing:
        return None, f"missing required fields: {missing}"

    fields: List[str] = []
    for field in ("instruction", "input", "output"):
        value = parsed.get(field)
        if isinstance(value, str):
            fields.append(value)
        elif value is None:
            fields.append("")
        else:
            fields.append(json.dumps(value, ensure_ascii=False))

    return (fields[0], fields[1], fields[2]), ""


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def localize_sample_with_retries(
    sample: Dict,
    target_lang: str,
    target_lang_name: str,
    args: argparse.Namespace,
    line_no: int,
) -> Tuple[str, str, str]:
    prompt = build_localization_prompt(
        sample=sample,
        target_lang=target_lang,
        target_lang_name=target_lang_name,
    )
    last_error = ""
    raw = ""

    for attempt in range(1, args.gemma_max_attempts + 1):
        query_prompt = prompt if attempt == 1 else build_retry_prompt(prompt, raw, last_error)

        def do_call_and_parse() -> Tuple[Tuple[str, str, str], str]:
            call_raw = query_gemma_server(
                prompt=query_prompt,
                endpoint=args.gemma_endpoint,
                timeout=args.gemma_timeout,
                max_tokens=args.gemma_max_tokens,
                temperature=args.gemma_temperature,
            )
            parsed = extract_first_json_object(call_raw)
            fields, error = validate_localized_fields(parsed)
            if fields is None:
                raw_preview = truncate_text(call_raw, args.error_raw_chars)
                raise ValueError(error + f"; raw={raw_preview}")
            return fields, call_raw

        try:
            fields, raw = call_with_backoff(
                do_call_and_parse,
                max_retries=args.gemma_backoff_retries,
                base_sleep=args.gemma_backoff_base_sleep,
            )
            if attempt > 1:
                print(f"[info] line={line_no} lang={target_lang} recovered valid JSON on attempt {attempt}")
            return fields
        except Exception as e:
            last_error = str(e)
            print(
                f"[warn] line={line_no} lang={target_lang} invalid Gemma JSON "
                f"on prompt attempt {attempt}/{args.gemma_max_attempts}: {last_error}"
            )

    raise RuntimeError(
        f"Line {line_no}, lang={target_lang}: Gemma did not return usable JSON "
        f"after {args.gemma_max_attempts} attempts. Last error: {last_error}"
    )


def localize_for_language(args: argparse.Namespace, target_lang: str) -> None:
    target_lang = target_lang.lower()
    target_lang_name = LANG_NAME_BY_TAG.get(target_lang, target_lang)

    out_dir = os.path.join(args.output_root, target_lang)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output_filename)
    state_path = out_path + ".state.json"
    quarantine_dir = os.path.join(out_dir, "quarantine")
    quarantine_path = os.path.join(
        quarantine_dir,
        os.path.splitext(args.output_filename)[0] + "_quarantine.jsonl",
    )

    if args.resume and os.path.exists(out_path):
        already_done = load_resume_state(state_path)
        if already_done <= 0:
            already_done = count_nonempty_lines(out_path)
        mode = "a"
        quarantine_mode = "a"
        print(f"[info] Resume enabled for lang={target_lang}. Skipping input lines <= {already_done}.")
    else:
        already_done = 0
        mode = "w"
        quarantine_mode = "w"

    total_records = count_nonempty_lines(args.input)
    total_to_process = max(0, total_records - already_done)
    if args.max_records > 0:
        total_to_process = min(total_to_process, args.max_records)

    pbar = None
    if not args.disable_tqdm and tqdm is not None:
        pbar = tqdm(total=total_to_process, desc=target_lang, unit="sample", dynamic_ncols=True)
    elif not args.disable_tqdm and tqdm is None:
        print("[warn] tqdm is not installed; progress bar disabled.")

    processed = 0
    written = 0
    quarantined = 0
    os.makedirs(quarantine_dir, exist_ok=True)
    with open(out_path, mode, encoding="utf-8") as out_f, open(quarantine_path, quarantine_mode, encoding="utf-8") as quarantine_f:
        for line_no, sample in iter_jsonl(args.input):
            if line_no <= already_done:
                continue
            if args.max_records > 0 and processed >= args.max_records:
                break

            try:
                instruction, inp, output = localize_sample_with_retries(
                    sample=sample,
                    target_lang=target_lang,
                    target_lang_name=target_lang_name,
                    args=args,
                    line_no=line_no,
                )
            except Exception as e:
                quarantined += 1
                quarantine_record = {
                    "line_no": line_no,
                    "sample_id": resolve_sample_id(sample, fallback_line_no=line_no),
                    "language": target_lang,
                    "error": str(e),
                    "source_sample": sample,
                }
                quarantine_f.write(json.dumps(quarantine_record, ensure_ascii=False) + "\n")
                quarantine_f.flush()
                processed += 1
                save_resume_state(state_path, line_no)
                if pbar is not None:
                    pbar.update(1)
                print(
                    f"[warn] line={line_no} lang={target_lang} quarantined after failed localization. "
                    f"quarantine={quarantine_path}"
                )
                continue

            localized = {
                "sample_id": resolve_sample_id(sample, fallback_line_no=line_no),
                "instruction": instruction,
                "input": inp,
                "output": output,
                "language": target_lang,
            }
            for key in ("primary_task", "secondary_tasks", "primary_domain", "secondary_domains"):
                if key in sample:
                    localized[key] = sample[key]

            out_f.write(json.dumps(localized, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
            processed += 1
            save_resume_state(state_path, line_no)
            if pbar is not None:
                pbar.update(1)
            if processed % args.log_every == 0:
                print(
                    f"[info] lang={target_lang} processed={processed} "
                    f"written={written} quarantined={quarantined} out={out_path}"
                )

    if pbar is not None:
        pbar.close()

    print(
        f"[done] lang={target_lang} processed={processed} "
        f"written={written} quarantined={quarantined} out={out_path}"
    )
    if quarantined:
        print(f"[done] lang={target_lang} quarantine file: {quarantine_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use Gemma to create target-language counterparts of discarded Alpaca samples."
    )
    parser.add_argument(
        "--input",
        default="/mnt/mirror1/aleixsant/alpaca_cleaned/en/alpaca_discarted_samples.jsonl",
        help="English discarded samples JSONL.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/mirror1/aleixsant/alpaca_cleaned",
        help="Root output dir. Outputs go to <output-root>/<lang>/<output-filename>.",
    )
    parser.add_argument(
        "--output-filename",
        default="alpaca_discarted_samples_localized.jsonl",
        help="Output filename inside each language folder.",
    )
    parser.add_argument("--target-langs", nargs="+", required=True, help="Target language tags.")
    parser.add_argument("--resume", action="store_true", help="Append and skip already-written records.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap per target language.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--gemma-endpoint", default="http://127.0.0.1:10001/chat/completions")
    parser.add_argument("--gemma-timeout", type=int, default=130)
    parser.add_argument("--gemma-max-tokens", type=int, default=4096)
    parser.add_argument("--gemma-max-attempts", type=int, default=3)
    parser.add_argument("--gemma-backoff-retries", type=int, default=2)
    parser.add_argument("--gemma-backoff-base-sleep", type=float, default=0.8)
    parser.add_argument("--gemma-temperature", type=float, default=0.0)
    parser.add_argument(
        "--error-raw-chars",
        type=int,
        default=800,
        help="Max raw Gemma response chars stored in quarantine errors. Use 0 for no truncation.",
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.gemma_max_attempts < 1:
        raise ValueError("--gemma-max-attempts must be >= 1")
    if args.gemma_backoff_retries < 1:
        raise ValueError("--gemma-backoff-retries must be >= 1")
    if args.gemma_backoff_base_sleep < 0:
        raise ValueError("--gemma-backoff-base-sleep must be >= 0")

    print(f"[info] Input: {args.input}")
    print(f"[info] Output root: {args.output_root}")
    print(f"[info] Output filename: {args.output_filename}")
    print(f"[info] Target languages: {' '.join(args.target_langs)}")
    print(f"[info] Gemma endpoint: {args.gemma_endpoint}")
    print(f"[info] Gemma max prompt attempts: {args.gemma_max_attempts}")
    print(f"[info] Gemma backoff retries per prompt: {args.gemma_backoff_retries}")
    print(f"[info] Resume mode: {args.resume}")

    for target_lang in args.target_langs:
        localize_for_language(args, target_lang)


if __name__ == "__main__":
    main()
