import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "data-pipeline" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from alpaca_localization.assemble_prevalidation_dataset import load_index
from alpaca_localization.languages import (
    LANG_NAME_BY_TAG,
    NLLB_LANG_BY_TAG,
    TARGET_LANGS,
)
from alpaca_localization.localize_discarded_samples import is_localization_candidate
from alpaca_localization.merge_validated_outputs import pick_field_value
from alpaca_localization.prepare_source import normalize_record, read_records


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


class RoutingAndMergeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
