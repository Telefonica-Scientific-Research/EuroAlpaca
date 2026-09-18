#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from .languages import NLLB_LANG_BY_TAG, TARGET_LANGS


DEFAULT_TARGET_LANGS = list(TARGET_LANGS)


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


def load_resume_state(path: str) -> Tuple[int, int]:
    if not os.path.exists(path):
        return 0, -1
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        last_input_line = obj.get("last_input_line", 0)
        output_offset = obj.get("output_offset", -1)
        if not isinstance(last_input_line, int) or not isinstance(output_offset, int):
            return 0, -1
        return last_input_line, output_offset
    except Exception:
        return 0, -1


def save_resume_state(path: str, last_input_line: int, output_offset: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_input_line": int(last_input_line),
                "output_offset": int(output_offset),
            },
            f,
            ensure_ascii=False,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    state_dir = os.path.dirname(os.path.abspath(path))
    dir_fd = os.open(state_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def commit_resume_state(out_f, state_path: str, last_input_line: int) -> None:
    """Durably write output before advancing its resume checkpoint."""
    out_f.flush()
    os.fsync(out_f.fileno())
    save_resume_state(state_path, last_input_line, out_f.tell())


def prepare_resume_output(out_path: str, state_path: str) -> int:
    """Restore output to its last committed offset and return the input line."""
    last_input_line, output_offset = load_resume_state(state_path)
    if last_input_line <= 0 or output_offset < 0:
        raise RuntimeError(
            f"Cannot safely resume {out_path}: missing, invalid, or legacy state file "
            f"{state_path}. Move the partial output and state aside or rerun without --resume."
        )

    output_size = os.path.getsize(out_path)
    if output_offset > output_size:
        raise RuntimeError(
            f"Cannot safely resume {out_path}: checkpoint offset {output_offset} "
            f"exceeds output size {output_size}."
        )

    # Output written after the last checkpoint was never committed. Removing it
    # prevents duplicate rows when those input lines are processed again.
    with open(out_path, "r+b") as out_f:
        out_f.truncate(output_offset)
        out_f.flush()
        os.fsync(out_f.fileno())
    return last_input_line


def get_sample_include(sample: Dict, sample_policy_key: str) -> bool:
    v = sample.get(sample_policy_key, True)
    if isinstance(v, bool):
        return v
    return True


def get_field_should_translate(
    sample: Dict,
    target_lang: str,
    field: str,
    field_policy_key: str,
) -> bool:
    value = sample.get(field, "")
    if not isinstance(value, str) or not value.strip():
        return False

    policy_root = sample.get(field_policy_key)
    if not isinstance(policy_root, dict):
        return True

    field_policy = policy_root.get(field)
    if not isinstance(field_policy, dict):
        return True

    all_obj = field_policy.get("all_languages")
    should_translate = True
    if isinstance(all_obj, dict) and isinstance(all_obj.get("should_translate"), bool):
        should_translate = bool(all_obj["should_translate"])

    per_obj = field_policy.get("per_language")
    if isinstance(per_obj, dict):
        override = per_obj.get(target_lang.lower()) or per_obj.get(target_lang)
        if isinstance(override, dict) and isinstance(override.get("should_translate"), bool):
            should_translate = bool(override["should_translate"])

    return should_translate


class NLLBTranslator:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.args = args
        self.torch = torch

        dtype = torch.float16
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float32":
            dtype = torch.float32

        if dtype == torch.bfloat16 and isinstance(args.device_map, str) and args.device_map.startswith("cuda"):
            bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            if not bf16_ok:
                print("[warn] BF16 not supported on this CUDA device; falling back to float16.")
                dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map=args.device_map,
        )
        self.model.eval()

        src_tag = args.source_lang.lower()
        self.source_code = NLLB_LANG_BY_TAG.get(src_tag, "")
        if not self.source_code:
            raise RuntimeError(f"Missing NLLB source code for {args.source_lang}")

        self._target_cache: Dict[str, Tuple[str, int]] = {}

    def _resolve_target(self, target_lang: str) -> Tuple[str, int]:
        t = target_lang.lower()
        if t in self._target_cache:
            return self._target_cache[t]

        target_code = NLLB_LANG_BY_TAG.get(t, "")
        if not target_code:
            raise RuntimeError(f"Missing NLLB target code for {target_lang}")

        tok = self.tokenizer
        if hasattr(tok, "lang_code_to_id") and target_code in tok.lang_code_to_id:
            forced_bos_token_id = tok.lang_code_to_id[target_code]
        else:
            forced_bos_token_id = tok.convert_tokens_to_ids(target_code)
        if not isinstance(forced_bos_token_id, int) or forced_bos_token_id < 0:
            raise RuntimeError(f"Could not resolve forced BOS token id for {target_code}")

        self._target_cache[t] = (target_code, forced_bos_token_id)
        return target_code, forced_bos_token_id

    def _translate_chunk(self, texts: List[str], target_lang: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        # Low-level translation unit:
        # - input: one chunk of unique texts for a single target language
        # - output: decoded strings + per-item diagnostics (same order/length)
        if not texts:
            return [], []

        # Resolve target language once to the NLLB BOS token used by generate().
        # target_code is also a validation checkpoint that mapping exists.
        target_code, forced_bos_token_id = self._resolve_target(target_lang)
        tok = self.tokenizer
        model = self.model
        # NLLB requires explicit source language tag on tokenizer.
        tok.src_lang = self.source_code

        # First tokenization pass (no truncation/padding):
        # used only to record original source lengths and truncation risk.
        tokenized_inputs = tok(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        input_token_lengths = [len(ids) for ids in tokenized_inputs]
        # Flag records whether this sample will be truncated in the model pass.
        input_truncated_flags = [n > self.args.max_input_tokens for n in input_token_lengths]

        # Second tokenization pass (actual model input):
        # truncates to max_input_tokens and pads to a rectangular tensor batch.
        encoded = tok(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_input_tokens,
            padding=True,
        )
        # Keep tensors on the same device as the loaded model shard.
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        # Adaptive decoding budget based on longest source in this chunk.
        # This prevents very short outputs for long inputs while still capping
        # generation cost via min/max bounds.
        longest = max(int(v.shape[-1]) for v in encoded["input_ids"])
        adaptive = int(longest * self.args.max_new_tokens_ratio) + self.args.max_new_tokens_extra
        max_new_tokens = max(self.args.min_new_tokens, min(self.args.max_new_tokens, adaptive))

        # Pure inference path; no gradients needed for translation.
        with self.torch.inference_mode():
            out = model.generate(
                **encoded,
                # Force generation to begin in the requested target language.
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=self.args.num_beams,
                repetition_penalty=self.args.repetition_penalty,
                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            )
        # Strip special tokens so output is plain text.
        decoded = tok.batch_decode(out, skip_special_tokens=True)

        # Compute output token lengths for diagnostics.
        # If pad token exists, exclude pads from the token count.
        pad_id = tok.pad_token_id
        if pad_id is None:
            output_token_lengths = [int(seq.shape[-1]) for seq in out]
        else:
            output_token_lengths = [int((seq != pad_id).sum().item()) for seq in out]
        # Hitting cap is a useful warning signal for possible truncation.
        output_hit_cap_flags = [n >= max_new_tokens for n in output_token_lengths]

        # Per-item diagnostics aligned to decoded[] index.
        diagnostics: List[Dict[str, Any]] = []
        for i in range(len(decoded)):
            diagnostics.append(
                {
                    "input_tokens": int(input_token_lengths[i]),
                    "max_input_tokens": int(self.args.max_input_tokens),
                    "input_truncated": bool(input_truncated_flags[i]),
                    "output_tokens": int(output_token_lengths[i]),
                    "max_new_tokens": int(max_new_tokens),
                    "output_hit_cap": bool(output_hit_cap_flags[i]),
                }
            )

        return decoded, diagnostics

    def translate(
        self,
        texts: List[str],
        target_lang: str,
        text_batch_size: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        # Batch-level deduplication for efficiency.
        # Repeated strings are translated once, then reused.
        unique_texts: List[str] = []
        text_to_idx: Dict[str, int] = {}
        # For each original text position, store index into unique_texts.
        text_indices: List[int] = []

        for t in texts:
            idx = text_to_idx.get(t)
            if idx is None:
                idx = len(unique_texts)
                unique_texts.append(t)
                text_to_idx[t] = idx
            text_indices.append(idx)

        # Hold results in unique-text space first.
        unique_translations = ["" for _ in unique_texts]
        unique_diagnostics: List[Dict[str, Any]] = [{} for _ in unique_texts]
        # Translate unique texts in chunks to control memory/latency.
        for start in range(0, len(unique_texts), text_batch_size):
            chunk = unique_texts[start : start + text_batch_size]
            translated, diagnostics = self._translate_chunk(chunk, target_lang)
            # Hard sanity checks to avoid silent index drift/corruption.
            if len(translated) != len(chunk):
                raise RuntimeError("Translation count mismatch")
            if len(diagnostics) != len(chunk):
                raise RuntimeError("Diagnostics count mismatch")
            for i, tr in enumerate(translated):
                unique_translations[start + i] = tr
                unique_diagnostics[start + i] = diagnostics[i]

        # Remap unique-space outputs back to original input order,
        # preserving duplicates exactly where they appeared.
        final_translations = [unique_translations[idx] for idx in text_indices]
        final_diagnostics = [unique_diagnostics[idx] for idx in text_indices]
        return final_translations, final_diagnostics


def flush_batch(
    out_f,
    batch_samples: List[Dict],
    jobs: List[Tuple[int, str, str, str]],
    translator: NLLBTranslator,
    target_lang: str,
    text_batch_size: int,
    warn_state: Dict[str, int],
    warn_limit_per_lang: int,
) -> Tuple[int, int, int]:
    truncated_fields = 0
    output_cap_fields = 0

    if jobs:
        texts = [j[2] for j in jobs]
        translated, diagnostics = translator.translate(
            texts,
            target_lang=target_lang,
            text_batch_size=text_batch_size,
        )
        if len(translated) != len(jobs):
            raise RuntimeError("Translated count mismatch in flush_batch")
        if len(diagnostics) != len(jobs):
            raise RuntimeError("Diagnostics count mismatch in flush_batch")
        for (sample_idx, field_name, _src, tag), tr, diag in zip(jobs, translated, diagnostics):
            batch_samples[sample_idx][field_name] = tr
            if diag.get("input_truncated"):
                truncated_fields += 1
            if diag.get("output_hit_cap"):
                output_cap_fields += 1

            if warn_state["printed"] < warn_limit_per_lang and (
                diag.get("input_truncated") or diag.get("output_hit_cap")
            ):
                reasons: List[str] = []
                if diag.get("input_truncated"):
                    reasons.append(
                        f"input_truncated({diag.get('input_tokens')}>{diag.get('max_input_tokens')})"
                    )
                if diag.get("output_hit_cap"):
                    reasons.append(
                        f"output_hit_cap({diag.get('output_tokens')}>={diag.get('max_new_tokens')})"
                    )
                print(f"[warn] lang={target_lang} {tag} {'; '.join(reasons)}")
                warn_state["printed"] += 1

    for sample in batch_samples:
        out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(batch_samples), truncated_fields, output_cap_fields


def main() -> None:
    # This entrypoint translates an already policy-annotated EN JSONL into one
    # JSONL per target language. Translation is constrained by:
    # - sample-level gate: sample_translation_policy
    # - field-level gate:  field_translation_policy
    parser = argparse.ArgumentParser(
        description=(
            "Translate annotated Alpaca JSONL using facebook/nllb-200-3.3B, "
            "respecting sample_translation_policy and field_translation_policy."
        )
    )
    # Data locations.
    parser.add_argument(
        "--input",
        required=True,
        help="Annotated input JSONL path.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root output dir. Outputs go to <output-root>/<lang>/<output-filename>.",
    )
    parser.add_argument(
        "--output-filename",
        default="alpaca_cleaned_labeled_translated.jsonl",
        help="Output filename inside each language folder.",
    )
    # Language/model settings.
    parser.add_argument("--source-lang", default="en", help="Source language tag.")
    parser.add_argument(
        "--target-langs",
        nargs="+",
        default=DEFAULT_TARGET_LANGS,
        help="Target language tags.",
    )
    parser.add_argument("--model-id", default="facebook/nllb-200-3.3B", help="HF model id.")
    parser.add_argument("--device-map", default="cuda:0", help="Device map for transformers.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision.",
    )
    # Translation/control settings.
    parser.add_argument("--fields", nargs="+", default=["instruction", "input", "output"])
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="If > 0, process at most this many non-empty input records per target language.",
    )
    parser.add_argument("--record-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-new-tokens", type=int, default=12)
    parser.add_argument("--max-new-tokens-ratio", type=float, default=1.25)
    parser.add_argument("--max-new-tokens-extra", type=int, default=24)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="Resume from per-language state file.")
    parser.add_argument(
        "--state-every",
        type=int,
        default=500,
        help="Persist resume state every N processed input records.",
    )
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--warn-limit-per-lang",
        type=int,
        default=40,
        help="Maximum number of per-field truncation/cap warnings printed per language.",
    )
    parser.add_argument("--sample-policy-key", default="sample_translation_policy")
    parser.add_argument("--field-policy-key", default="field_translation_policy")
    parser.add_argument(
        "--keep-discarded",
        action="store_true",
        help="Keep sample_translation_policy=false records in output unchanged.",
    )
    parser.add_argument(
        "--drop-policy-fields",
        action="store_true",
        help="Drop annotation policy fields from output records.",
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars.")

    # Parse once; all runtime behavior reads from args.
    args = parser.parse_args()

    # Fail early if input is missing.
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    # Used for progress bars (same input is scanned once per target language).
    total_records = count_nonempty_lines(args.input)

    # Load NLLB model/tokenizer once and reuse across all target languages.
    translator = NLLBTranslator(args)

    # Process each target language independently into its own output file.
    for target_lang in [t.lower() for t in args.target_langs]:
        # Skip no-op translation (source == target).
        if target_lang == args.source_lang.lower():
            print(f"[warn] Skipping target_lang={target_lang} because it matches source language.")
            continue
        # Ensure there is a mapping to an NLLB language code.
        if target_lang not in NLLB_LANG_BY_TAG:
            raise RuntimeError(f"Unknown target language tag for NLLB mapping: {target_lang}")

        # Each target language writes to <output-root>/<lang>/<filename>.
        out_dir = os.path.join(args.output_root, target_lang)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.output_filename)
        # Resume checkpoint stores the committed input line and output offset.
        state_path = out_path + ".state.json"

        # Resume appends output and skips already-processed input lines.
        if args.resume and os.path.exists(out_path):
            mode = "a"
            already_line = prepare_resume_output(out_path, state_path)
            print(
                f"[info] Resume enabled for lang={target_lang}. "
                f"Skipping input lines <= {already_line}."
            )
        else:
            mode = "w"
            already_line = 0
            # A new output must never inherit a checkpoint from an older run.
            if os.path.exists(state_path):
                os.remove(state_path)

        # Per-language counters and in-memory batch buffers.
        processed_input = 0
        written_output = 0
        dropped_samples = 0
        truncated_fields = 0
        output_cap_fields = 0
        # batch_samples: output records waiting to be written.
        batch_samples: List[Dict] = []
        # jobs: translation work items (sample index + field + text + debug tag).
        jobs: List[Tuple[int, str, str, str]] = []
        last_input_line = already_line
        # Shared warning limiter for verbose truncation/cap warnings.
        warn_state = {"printed": 0}

        def flush_pending_batch(out_f) -> None:
            nonlocal batch_samples, jobs
            nonlocal written_output, truncated_fields, output_cap_fields
            if not batch_samples:
                return
            written_inc, trunc_inc, cap_inc = flush_batch(
                out_f=out_f,
                batch_samples=batch_samples,
                jobs=jobs,
                translator=translator,
                target_lang=target_lang,
                text_batch_size=args.text_batch_size,
                warn_state=warn_state,
                warn_limit_per_lang=args.warn_limit_per_lang,
            )
            written_output += written_inc
            truncated_fields += trunc_inc
            output_cap_fields += cap_inc
            batch_samples = []
            jobs = []

        pbar = None
        # Progress bar tracks input rows processed for this target language.
        if not args.disable_tqdm and tqdm is not None:
            pbar = tqdm(
                total=max(0, total_records - already_line),
                desc=f"{target_lang}",
                unit="sample",
                dynamic_ncols=True,
            )

        with open(out_path, mode, encoding="utf-8") as out_f:
            # Re-scan input from the beginning; resume skip handles checkpointing.
            for line_no, sample in iter_jsonl(args.input):
                if line_no <= already_line:
                    continue
                # The cap counts input records regardless of their routing. Check
                # it before processing so an excluded boundary record cannot
                # bypass the limit via the continue below.
                if args.max_records > 0 and processed_input >= args.max_records:
                    break

                processed_input += 1
                last_input_line = line_no
                if pbar is not None:
                    pbar.update(1)

                # Sample-level gate: include/exclude this sample from localization.
                include_sample = get_sample_include(sample, args.sample_policy_key)
                # Excluded samples are dropped unless keep_discarded is requested.
                if not include_sample and not args.keep_discarded:
                    dropped_samples += 1
                    # A checkpoint may advance past dropped rows only after any
                    # earlier buffered translations have been written.
                    if args.resume and processed_input % args.state_every == 0:
                        flush_pending_batch(out_f)
                        commit_resume_state(out_f, state_path, last_input_line)
                    continue

                # Work on a copy; never mutate the input sample object directly.
                out_sample = dict(sample)
                if args.drop_policy_fields:
                    # Optional cleanup: remove policy metadata from translated output.
                    out_sample.pop(args.sample_policy_key, None)
                    out_sample.pop(args.field_policy_key, None)

                # Position of this sample inside the current record batch.
                sample_idx = len(batch_samples)

                if include_sample:
                    # Build field translation jobs only for allowed fields.
                    for field in args.fields:
                        val = out_sample.get(field, "")
                        # Skip empty/non-string fields.
                        if not isinstance(val, str) or not val.strip():
                            continue
                        # Field-level gate (all_languages default + per-language overrides).
                        if get_field_should_translate(
                            sample=sample,
                            target_lang=target_lang,
                            field=field,
                            field_policy_key=args.field_policy_key,
                        ):
                            # Store minimal job payload; actual translation happens in flush_batch.
                            jobs.append((sample_idx, field, val, f"line={line_no},field={field}"))

                # Keep sample in batch whether translated or unchanged.
                batch_samples.append(out_sample)

                # Flush when record batch reaches threshold.
                if len(batch_samples) >= args.record_batch_size:
                    flush_pending_batch(out_f)

                # Periodic language-level progress logging.
                if processed_input % args.log_every == 0:
                    print(
                        f"[info] lang={target_lang} processed_input={processed_input} "
                        f"written={written_output} dropped={dropped_samples} "
                        f"input_truncated_fields={truncated_fields} "
                        f"output_cap_fields={output_cap_fields}"
                    )

                # Periodic resume checkpoint.
                if args.resume and processed_input % args.state_every == 0:
                    flush_pending_batch(out_f)
                    commit_resume_state(out_f, state_path, last_input_line)

            # Flush tail batch (if any) after loop ends.
            flush_pending_batch(out_f)
            # The final checkpoint is committed while the output handle is open,
            # after every remaining row has been written and made durable.
            if args.resume:
                commit_resume_state(out_f, state_path, last_input_line)
        if pbar is not None:
            pbar.close()

        # If warning print cap was reached, remind user that totals are below.
        if warn_state["printed"] >= args.warn_limit_per_lang and (
            truncated_fields > 0 or output_cap_fields > 0
        ):
            print(
                f"[info] lang={target_lang} warning print limit reached "
                f"({args.warn_limit_per_lang}); see totals below."
            )

        # Final summary for this target language.
        print(
            f"[done] lang={target_lang} input_processed={processed_input} "
            f"written={written_output} dropped={dropped_samples} "
            f"input_truncated_fields={truncated_fields} "
            f"output_cap_fields={output_cap_fields} out={out_path}"
        )


if __name__ == "__main__":
    main()
