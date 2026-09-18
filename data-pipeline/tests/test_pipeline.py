import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "data-pipeline" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from alpaca_localization.assemble_prevalidation_dataset import load_index
from alpaca_localization.translate_with_nllb import (
    commit_resume_state,
    load_resume_state,
    prepare_resume_output,
)
from alpaca_localization.languages import (
    LANG_NAME_BY_TAG,
    NLLB_LANG_BY_TAG,
    TARGET_LANGS,
)
from alpaca_localization import localize_discarded_samples as localization
from alpaca_localization import translate_with_nllb as translation
from alpaca_localization.localize_discarded_samples import is_localization_candidate
from alpaca_localization.merge_validated_outputs import (
    build_validated_index,
    pick_field_value,
)
from alpaca_localization.prepare_source import normalize_record, read_records
from alpaca_localization.split_by_primary_task import (
    ALL_TASK_TYPES as SPLIT_TASK_TYPES,
)
from alpaca_localization.translation_policy_annotation.prompts import (
    build_sample_policy_prompt,
)
from alpaca_localization import validate_translation_coherence as validation


class LanguageConfigurationTest(unittest.TestCase):
    def test_exactly_fifty_target_languages_have_nllb_codes(self):
        self.assertEqual(50, len(TARGET_LANGS))
        self.assertEqual(50, len(set(TARGET_LANGS)))
        self.assertEqual(set(TARGET_LANGS), set(LANG_NAME_BY_TAG))
        self.assertTrue(set(TARGET_LANGS).issubset(NLLB_LANG_BY_TAG))
        self.assertEqual("eng_Latn", NLLB_LANG_BY_TAG["en"])

    def test_shell_and_python_language_lists_match(self):
        command = (
            'source data-pipeline/scripts/languages.sh; '
            'printf "%s\\n" "${EUROALPACA_LANGS[@]}"'
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(list(TARGET_LANGS), result.stdout.splitlines())


class SourcePreparationTest(unittest.TestCase):
    def test_reads_json_array_and_normalizes_records(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(
                json.dumps(
                    [
                        {"instruction": "Do it", "input": None, "output": "Done"},
                        {"instruction": "Again", "input": "x", "output": "y"},
                    ]
                ),
                encoding="utf-8",
            )
            records = list(read_records(source))

        self.assertEqual(2, len(records))
        normalized = normalize_record(records[0], 1)
        self.assertEqual("", normalized["input"])
        self.assertEqual(1, normalized["sample_id"])

    def test_rejects_non_string_text_fields(self):
        with self.assertRaises(ValueError):
            normalize_record(
                {"instruction": ["bad"], "input": "", "output": ""},
                1,
            )


class PromptAlignmentTest(unittest.TestCase):
    def test_semantic_annotation_uses_paper_task_label(self):
        annotation_source = (
            SOURCE_ROOT / "alpaca_localization" / "annotate_tasks_domains.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"summarisation"', annotation_source)
        self.assertNotIn('"summarization"', annotation_source)
        self.assertIn("summarisation", SPLIT_TASK_TYPES)
        self.assertNotIn("summarization", SPLIT_TASK_TYPES)

    def test_sample_gate_uses_paper_wording(self):
        prompt = build_sample_policy_prompt(
            {
                "instruction": "Summarise this",
                "input": "Text",
                "output": "Summary",
                "primary_task": "summarisation",
                "primary_domain": "general",
                "secondary_tasks": [],
                "secondary_domains": [],
            },
            source_lang_name="English",
            target_langs=["de"],
        )
        self.assertIn(
            "can be preserved through field-wise machine translation",
            prompt,
        )
        self.assertIn("Target languages: de (German)", prompt)


class ValidationRetryTest(unittest.TestCase):
    def test_invalid_response_is_repaired_on_next_prompt_attempt(self):
        args = SimpleNamespace(
            gemma_max_attempts=2,
            gemma_backoff_retries=1,
            gemma_backoff_base_sleep=0.0,
            gemma_endpoint="unused",
            gemma_timeout=1,
            gemma_max_tokens=100,
            gemma_temperature=0.0,
            error_raw_chars=1000,
        )
        valid = json.dumps(
            {"is_coherent": True, "issues": "", "changes": {}}
        )

        with mock.patch.object(
            validation,
            "query_gemma_server",
            side_effect=["not JSON", valid],
        ) as query:
            parsed, _raw, error = validation.query_validation_with_retries(
                prompt="original validation prompt",
                args=args,
                line_no=1,
                language_tag="de",
            )

        self.assertEqual("", error)
        self.assertTrue(parsed["is_coherent"])
        self.assertEqual(2, query.call_count)
        self.assertIn(
            "Your previous response could not be used",
            query.call_args_list[1].kwargs["prompt"],
        )

    def test_validation_schema_rejects_incomplete_json(self):
        self.assertIn("missing required fields", validation.validate_response_schema({}))

    def test_summary_distinguishes_written_and_quarantined_records(self):
        valid = {"is_coherent": True, "issues": "", "changes": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "output.jsonl"
            source.write_text(
                '{"sample_id": 1, "instruction": "a", "input": "", "output": "b"}\n'
                '{"sample_id": 2, "instruction": "c", "input": "", "output": "d"}\n',
                encoding="utf-8",
            )
            argv = [
                "validate_translation_coherence",
                "--input",
                str(source),
                "--output",
                str(output),
                "--language-tag",
                "de",
                "--disable-tqdm",
            ]
            with mock.patch.object(
                validation,
                "query_validation_with_retries",
                side_effect=[(None, "bad", "invalid"), (valid, "valid", "")],
            ), mock.patch.object(sys, "argv", argv), io.StringIO() as stdout:
                with contextlib.redirect_stdout(stdout):
                    validation.main()
                summary = stdout.getvalue()

            self.assertIn("processed=2 written=1 quarantined=1", summary)
            self.assertEqual(1, len(output.read_text(encoding="utf-8").splitlines()))


class LocalizationRetryTest(unittest.TestCase):
    def test_localized_fields_require_strings_and_nonempty_required_text(self):
        invalid_rows = [
            {"instruction": None, "input": "", "output": "answer"},
            {"instruction": "task", "input": [], "output": "answer"},
            {"instruction": "task", "input": "", "output": {"bad": True}},
            {"instruction": "   ", "input": "", "output": "answer"},
            {"instruction": "task", "input": "", "output": "\t"},
        ]
        for row in invalid_rows:
            with self.subTest(row=row):
                fields, error = localization.validate_localized_fields(row)
                self.assertIsNone(fields)
                self.assertTrue(error)

        fields, error = localization.validate_localized_fields(
            {"instruction": "task", "input": "", "output": "answer"}
        )
        self.assertEqual(("task", "", "answer"), fields)
        self.assertEqual("", error)

    def test_repair_prompt_contains_the_invalid_response(self):
        args = SimpleNamespace(
            gemma_max_attempts=2,
            gemma_backoff_retries=1,
            gemma_backoff_base_sleep=0.0,
            gemma_endpoint="unused",
            gemma_timeout=1,
            gemma_max_tokens=100,
            gemma_temperature=0.0,
            error_raw_chars=1000,
        )
        invalid = json.dumps(
            {"instruction": ["wrong type"], "input": "", "output": "answer"}
        )
        valid = json.dumps(
            {"instruction": "Aufgabe", "input": "", "output": "Antwort"}
        )
        with mock.patch.object(
            localization,
            "query_gemma_server",
            side_effect=[invalid, valid],
        ) as query:
            fields = localization.localize_sample_with_retries(
                sample={"instruction": "Task", "input": "", "output": "Answer"},
                target_lang="de",
                target_lang_name="German",
                args=args,
                line_no=1,
            )

        self.assertEqual(("Aufgabe", "", "Antwort"), fields)
        self.assertEqual(2, query.call_count)
        self.assertIn(invalid, query.call_args_list[1].kwargs["prompt"])


class RoutingAndMergeTest(unittest.TestCase):
    def test_stage_four_max_records_counts_excluded_boundary_record(self):
        class FakeTranslator:
            def __init__(self, _args):
                pass

            def translate(self, texts, target_lang, text_batch_size):
                del target_lang, text_batch_size
                return list(texts), [
                    {"input_truncated": False, "output_hit_cap": False}
                    for _ in texts
                ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output_root = root / "output"
            rows = [
                {
                    "sample_id": 1,
                    "instruction": "one",
                    "input": "",
                    "output": "first",
                    "sample_translation_policy": True,
                },
                {
                    "sample_id": 2,
                    "instruction": "two",
                    "input": "",
                    "output": "second",
                    "sample_translation_policy": False,
                },
                {
                    "sample_id": 3,
                    "instruction": "three",
                    "input": "",
                    "output": "third",
                    "sample_translation_policy": True,
                },
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            argv = [
                "translate_with_nllb",
                "--input",
                str(source),
                "--output-root",
                str(output_root),
                "--target-langs",
                "de",
                "--max-records",
                "2",
                "--record-batch-size",
                "10",
                "--disable-tqdm",
            ]
            with mock.patch.object(translation, "NLLBTranslator", FakeTranslator), mock.patch.object(
                sys, "argv", argv
            ):
                translation.main()

            output_rows = [
                json.loads(line)
                for line in (output_root / "de" / "alpaca_cleaned_labeled_translated.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([1], [row["sample_id"] for row in output_rows])

    def test_resume_discards_output_after_last_durable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "translated.jsonl"
            state = Path(str(output) + ".state.json")
            committed = '{"sample_id": 1, "text": "Grüße"}\n'
            with output.open("w", encoding="utf-8") as out_f:
                out_f.write(committed)
                commit_resume_state(out_f, str(state), 1)
                out_f.write('{"sample_id": 2}\n')
                out_f.flush()
                os.fsync(out_f.fileno())

            self.assertEqual((1, len(committed.encode("utf-8"))), load_resume_state(str(state)))
            self.assertEqual(1, prepare_resume_output(str(output), str(state)))
            self.assertEqual(committed, output.read_text(encoding="utf-8"))

    def test_only_false_policy_routes_to_task_localization(self):
        self.assertTrue(is_localization_candidate({"route": False}, "route", False))
        self.assertFalse(is_localization_candidate({"route": True}, "route", False))
        self.assertFalse(is_localization_candidate({}, "route", False))
        self.assertTrue(is_localization_candidate({}, "route", True))

    def test_load_index_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                '{"sample_id": 1, "instruction": "a"}\n'
                '{"sample_id": 1, "instruction": "b"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_index(str(path), "test")

    def test_stage_eight_rejects_duplicate_validated_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validated.jsonl"
            path.write_text(
                '{"sample_id": 1}\n{"sample_id": 1}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate sample_id"):
                build_validated_index(str(path))

    def test_stage_eight_rejects_unexpected_validated_sample_ids(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.jsonl"
            validated = root / "validated.jsonl"
            output = root / "final.jsonl"
            original.write_text(
                '{"sample_id": 1, "instruction": "a", "input": "", "output": "b"}\n',
                encoding="utf-8",
            )
            validated.write_text(
                '{"sample_id": 2, "instruction": "c", "input": "", "output": "d"}\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alpaca_localization.merge_validated_outputs",
                    "--original",
                    str(original),
                    "--validated",
                    str(validated),
                    "--output",
                    str(output),
                    "--language",
                    "de",
                ],
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("absent from the original", result.stderr)
            self.assertFalse(output.exists())

    def test_validated_field_is_used_only_when_marked_changed(self):
        original = {"instruction": "original"}
        unchanged = {
            "instruction": "unexpected",
            "post_edit": {"instruction_changed": False},
        }
        changed = {
            "instruction": "corrected",
            "post_edit": {"instruction_changed": True},
        }
        self.assertEqual("original", pick_field_value("instruction", original, unchanged))
        self.assertEqual("corrected", pick_field_value("instruction", original, changed))

    def test_non_model_command_line_flow(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SOURCE_ROOT)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            normalized = root / "normalized.jsonl"
            translated = root / "translated.jsonl"
            localized = root / "localized.jsonl"
            assembled = root / "assembled.jsonl"
            validated = root / "validated.jsonl"
            final = root / "final.jsonl"

            source.write_text(
                json.dumps(
                    [
                        {"instruction": "one", "input": "", "output": "first"},
                        {"instruction": "two", "input": "", "output": "second"},
                    ]
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alpaca_localization.prepare_source",
                    "--input",
                    str(source),
                    "--output",
                    str(normalized),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

            normalized_rows = [
                json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()
            ]
            translated.write_text(
                json.dumps(normalized_rows[0], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            localized.write_text(
                json.dumps(normalized_rows[1], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alpaca_localization.assemble_prevalidation_dataset",
                    "--translated",
                    str(translated),
                    "--localized-discarded",
                    str(localized),
                    "--expected-source",
                    str(normalized),
                    "--output",
                    str(assembled),
                    "--language",
                    "de",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

            correction = dict(normalized_rows[0])
            correction["instruction"] = "eins"
            correction["post_edit"] = {
                "instruction_changed": True,
                "input_changed": False,
                "output_changed": False,
            }
            validated.write_text(
                json.dumps(correction, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alpaca_localization.merge_validated_outputs",
                    "--original",
                    str(assembled),
                    "--validated",
                    str(validated),
                    "--output",
                    str(final),
                    "--language",
                    "de",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

            final_rows = [
                json.loads(line) for line in final.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(2, len(final_rows))
            self.assertEqual("eins", final_rows[0]["instruction"])
            self.assertEqual("two", final_rows[1]["instruction"])
            self.assertTrue(all(row["language"] == "de" for row in final_rows))

    def test_assembly_rejects_missing_expected_sample(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "expected.jsonl"
            translated = root / "translated.jsonl"
            localized = root / "localized.jsonl"
            output = root / "assembled.jsonl"
            expected.write_text(
                '{"sample_id": 1}\n{"sample_id": 2}\n', encoding="utf-8"
            )
            translated.write_text('{"sample_id": 1}\n', encoding="utf-8")
            localized.write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alpaca_localization.assemble_prevalidation_dataset",
                    "--translated",
                    str(translated),
                    "--localized-discarded",
                    str(localized),
                    "--expected-source",
                    str(expected),
                    "--output",
                    str(output),
                    "--language",
                    "de",
                ],
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("missing=1", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
