"""CLI for the translation-policy annotator."""

import argparse
import json
import os

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from .constants import DEFAULT_TARGET_LANGS, LANG_NAME_BY_TAG
from .decisions import annotate_sample
from .io_utils import count_nonempty_lines, iter_jsonl

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Annotate each sample with compact translation policies for "
            "sample inclusion and instruction/input/output fields."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--output-sample-policy-field",
        default="sample_translation_policy",
        help="Field name where global sample-level include/exclude boolean is stored.",
    )
    parser.add_argument(
        "--output-sample-policy-reason-field",
        default="sample_translation_policy_reason",
        help="Field name where the final sample-level reason code is stored.",
    )
    parser.add_argument(
        "--output-sample-policy-evidence-field",
        default="sample_translation_policy_evidence",
        help="Field name where short evidence for the sample-level reason is stored.",
    )
    parser.add_argument(
        "--output-sample-policy-source-field",
        default="sample_translation_policy_source",
        help="Field name where the source of the final sample-level decision is stored.",
    )
    parser.add_argument(
        "--output-policy-field",
        default="field_translation_policy",
        help=(
            "Field name to store the model policy per field "
            "(all_languages default + per_language overrides)."
        ),
    )
    parser.add_argument("--source-lang", default="en", help="Source language tag.")
    parser.add_argument(
        "--target-langs",
        nargs="+",
        default=DEFAULT_TARGET_LANGS,
        help="Target language tags to annotate for.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for quick tests.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress print interval.")
    parser.add_argument("--gemma-endpoint", default="http://127.0.0.1:10001/chat/completions")
    parser.add_argument("--gemma-timeout", type=int, default=130)
    parser.add_argument("--gemma-max-tokens", type=int, default=260)
    parser.add_argument("--gemma-temperature", type=float, default=0.0)
    parser.add_argument(
        "--parse-failure-dir",
        default="",
        help=(
            "Directory for per-sample Gemma JSON parse failures. "
            "Default: <output>.parse_failures"
        ),
    )
    parser.add_argument(
        "--reuse-existing-policies-without-querying-gemma",
        action="store_true",
        dest="reuse_existing_policies_without_querying_gemma",
        help=(
            "If input already contains policy fields, reuse them as model output "
            "and only re-apply deterministic guardrails (no Gemma calls)."
        ),
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars.")
    return parser


def main() -> None:
    """Run the command-line annotation workflow."""
    args = build_arg_parser().parse_args()
    # Fail fast on missing input to avoid creating partial/empty outputs.
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    # Ensure output directory exists (safe for nested custom output paths).
    out_parent = os.path.dirname(os.path.abspath(args.output))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    # Resume mode: append to existing output and skip already-written input rows.
    # This assumes output order matches input order for previous successful writes.
    if args.resume and os.path.exists(args.output):
        already_done = count_nonempty_lines(args.output)
        mode = "a"
        print(f"[info] Resume enabled. Skipping first {already_done} records from input.")
    else:
        already_done = 0
        mode = "w"

    # Total work estimate used by tqdm. max_records limits *new* processing only.
    total_records = count_nonempty_lines(args.input)
    total_to_process = max(0, total_records - already_done)
    if args.max_records > 0:
        total_to_process = min(total_to_process, args.max_records)

    pbar = None
    # Progress bar is optional and does not affect logic; disabled if tqdm missing.
    if not args.disable_tqdm and tqdm is not None:
        pbar = tqdm(total=total_to_process, desc="decisions", unit="sample", dynamic_ncols=True)
    elif not args.disable_tqdm and tqdm is None:
        print("[warn] tqdm is not installed; progress bar disabled.")

    # Normalize language tags once so downstream lookups/comparisons are consistent.
    source_lang_name = LANG_NAME_BY_TAG.get(args.source_lang.lower())
    normalized_langs = [t.lower() for t in args.target_langs]
    # processed = newly handled rows in this invocation (not total file size).
    processed = 0
    # Count samples that stop at sample gate (no field policy generated).
    skipped_field_policy_samples = 0
    quarantined_parse_failures = 0
    if args.reuse_existing_policies_without_querying_gemma:
        print("[info] Reusing existing policy fields from input (Gemma calls disabled).")

    with open(args.output, mode, encoding="utf-8") as out_f:
        for idx, (_line_no, sample) in enumerate(iter_jsonl(args.input), start=1):
            # Resume skip by input index (1-based enumerate over full input).
            if idx <= already_done:
                continue
            # Optional short run for debugging/smoke tests.
            if args.max_records > 0 and processed >= args.max_records:
                break
            
            enriched, skipped_field_policy, parse_failure_count = annotate_sample(
                sample=sample,
                args=args,
                source_lang_name=source_lang_name,
                target_langs=normalized_langs,
            )
            if skipped_field_policy:
                skipped_field_policy_samples += 1
            quarantined_parse_failures += parse_failure_count
            out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            processed += 1
            if pbar is not None:
                # Advances the progress bar by 1 for each newly processed sample.
                pbar.update(1)
            # Periodic heartbeat: processed rows, skipped field policies, parse quarantines.
            if processed % args.log_every == 0:
                print(
                    f"[info] Processed {processed} samples... "
                    f"(field-policy skipped for {skipped_field_policy_samples}, "
                    f"parse failures quarantined: {quarantined_parse_failures})"
                )

    if pbar is not None:
        pbar.close()

    # Final summary for logs/checkpointing.
    print(
        f"[done] Wrote {processed} annotated records to: {args.output} "
        f"(field-policy skipped for {skipped_field_policy_samples} samples, "
        f"parse failures quarantined: {quarantined_parse_failures})"
    )

__all__ = ["build_arg_parser", "main"]
