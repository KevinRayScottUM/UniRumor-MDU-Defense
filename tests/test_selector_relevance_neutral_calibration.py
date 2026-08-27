from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.selector_relevance_neutral_calibration import build_neutral as neutral
from scripts.selector_relevance_neutral_calibration.build_neutral import (
    ExpectedCounts,
    NeutralBuildError,
    build_neutral_calibration,
    load_source_artifacts,
    neutralize_example,
    sha256_file,
)
from scripts.selector_relevance_neutral_calibration.run_build_neutral import build_parser


FIXTURE_COUNTS = ExpectedCounts(
    total_cases=2,
    total_examples=4,
    train_cases=1,
    dev_cases=1,
    ocr_examples=2,
    transcript_examples=2,
    dataset_case_counts=(("GroundLie360", 1), ("TRUE-3MFact", 1)),
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: list[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _candidate_units(case_token: str, positive_modality: str) -> list[Dict[str, Any]]:
    return [
        {
            "unit_id": f"{case_token}-ocr",
            "unit_type": "ocr",
            "modality": "ocr",
            "text": f"PUBLIC NOTICE FOR {case_token}",
            "relevance_target": int(positive_modality == neutral.OCR),
            "provenance": {"source": "locked-train", "order": 0},
        },
        {
            "unit_id": f"{case_token}-transcript",
            "unit_type": "transcript",
            "modality": "text",
            "text": f"The speaker discusses {case_token} in detail.",
            "relevance_target": int(positive_modality == neutral.TRANSCRIPT),
            "provenance": {"source": "locked-train", "order": 1},
        },
        {
            "unit_id": f"{case_token}-negative",
            "unit_type": "text",
            "modality": "text",
            "text": "A retained exposure candidate that is not directly grounded.",
            "relevance_target": 0,
            "provenance": {"source": "locked-train", "order": 2},
        },
    ]


def _example(
    case_token: str,
    dataset: str,
    split: str,
    modality: str,
) -> Dict[str, Any]:
    candidates = _candidate_units(case_token, modality)
    anchor = candidates[0] if modality == neutral.OCR else candidates[1]
    anchor_text = str(anchor["text"])
    prefix = neutral.OCR_PREFIX if modality == neutral.OCR else neutral.TRANSCRIPT_PREFIX
    return {
        "schema_version": 1,
        "calibration_example_id": f"{case_token}-{modality.casefold()}",
        "source_dataset": dataset,
        "source_case_id": f"{dataset}:train:{case_token}",
        "canonical_underlying_case_id": f"{dataset}:{case_token}",
        "calibration_split": split,
        "expected_modality": modality,
        "claim": prefix + anchor_text + neutral.CLAIM_SUFFIX,
        "anchor_unit_id": anchor["unit_id"],
        "anchor_text": anchor_text,
        "positive_unit_ids": [anchor["unit_id"]],
        "model_exposed_candidate_count": len(candidates),
        "candidate_units": candidates,
        "source_provenance": {
            "train_variant_sha256": "1" * 64,
            "phase4a_config_sha256": "2" * 64,
            "source_row_index": 7,
            "quality_policy": "closed-v2",
        },
    }


def _write_source_fixture(root: Path) -> Path:
    source = root / "source-calibration"
    source.mkdir()
    train = [
        _example("case-a", "GroundLie360", "train", neutral.OCR),
        _example("case-a", "GroundLie360", "train", neutral.TRANSCRIPT),
    ]
    dev = [
        _example("case-b", "TRUE-3MFact", "dev", neutral.OCR),
        _example("case-b", "TRUE-3MFact", "dev", neutral.TRANSCRIPT),
    ]
    split_manifest = {
        "schema_version": 1,
        "assignments": [
            {
                "canonical_underlying_case_id": "GroundLie360:case-a",
                "source_dataset": "GroundLie360",
                "calibration_split": "train",
            },
            {
                "canonical_underlying_case_id": "TRUE-3MFact:case-b",
                "source_dataset": "TRUE-3MFact",
                "calibration_split": "dev",
            },
        ],
    }
    eligible_manifest = {
        "schema_version": 1,
        "freeze_stage": "BEFORE_CALIBRATION_SPLIT_AND_TARGET_GENERATION",
        "eligible_case_count": 2,
        "eligible_cases": [
            {
                "source_dataset": "GroundLie360",
                "source_case_id": "GroundLie360:train:case-a",
                "canonical_underlying_case_id": "GroundLie360:case-a",
            },
            {
                "source_dataset": "TRUE-3MFact",
                "source_case_id": "TRUE-3MFact:train:case-b",
                "canonical_underlying_case_id": "TRUE-3MFact:case-b",
            },
        ],
    }
    payloads = {
        "calibration_train.jsonl": _jsonl_bytes(train),
        "calibration_dev.jsonl": _jsonl_bytes(dev),
        "calibration_split_manifest.json": _json_bytes(split_manifest),
        "eligible_case_manifest.json": _json_bytes(eligible_manifest),
    }
    hashes: Dict[str, str] = {}
    for name, content in payloads.items():
        (source / name).write_bytes(content)
        hashes[name] = _digest(content)
    for stem, suffix in (
        ("calibration_train", ".jsonl"),
        ("calibration_dev", ".jsonl"),
        ("calibration_split_manifest", ".json"),
        ("eligible_case_manifest", ".json"),
    ):
        (source / f"{stem}.sha256").write_text(
            hashes[f"{stem}{suffix}"] + "\n", encoding="utf-8"
        )
    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "implementation_revision": neutral.SOURCE_IMPLEMENTATION_REVISION,
        "eligible_case_count": 2,
        "calibration_train_case_count": 1,
        "calibration_dev_case_count": 1,
        "calibration_train_example_count": 2,
        "calibration_dev_example_count": 2,
        "ocr_example_count": 2,
        "transcript_example_count": 2,
        "dataset_case_counts": {"GroundLie360": 1, "TRUE-3MFact": 1},
        "dataset_split_counts": {
            "GroundLie360": {"train": 1, "dev": 0},
            "TRUE-3MFact": {"train": 0, "dev": 1},
        },
        "heldout_case_ids": ["GroundLie360:heldout", "TRUE-3MFact:heldout"],
        "artifact_sha256": hashes,
        "selection_outputs_inspected": False,
        "veracity_labels_inspected": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "training_started": False,
        "production_or_model_code_changed": False,
    }
    (source / "build_report.json").write_bytes(_json_bytes(report))
    return source


def _refresh_source_hash(source: Path, name: str) -> None:
    report_path = source / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = sha256_file(source / name)
    report["artifact_sha256"][name] = digest
    report_path.write_bytes(_json_bytes(report))
    stem = name.rsplit(".", 1)[0]
    sidecar = source / f"{stem}.sha256"
    if sidecar.exists():
        sidecar.write_text(digest + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class NeutralTransformationTests(unittest.TestCase):
    def test_ocr_and_transcript_use_the_exact_same_neutral_template(self) -> None:
        ocr = _example("case-a", "GroundLie360", "train", neutral.OCR)
        transcript = _example(
            "case-a", "GroundLie360", "train", neutral.TRANSCRIPT
        )
        neutral_ocr, ocr_manifest = neutralize_example(ocr)
        neutral_transcript, transcript_manifest = neutralize_example(transcript)
        self.assertEqual(
            'The relevant content states "PUBLIC NOTICE FOR case-a".',
            neutral_ocr["claim"],
        )
        self.assertEqual(
            'The relevant content states "The speaker discusses case-a in detail.".',
            neutral_transcript["claim"],
        )
        self.assertTrue(neutral_ocr["claim"].startswith(neutral.NEUTRAL_PREFIX))
        self.assertTrue(neutral_transcript["claim"].startswith(neutral.NEUTRAL_PREFIX))
        for row in (ocr_manifest, transcript_manifest):
            self.assertTrue(row["anchor_text_unchanged"])
            self.assertTrue(row["candidate_ids_unchanged"])
            self.assertTrue(row["candidate_order_unchanged"])
            self.assertTrue(row["candidate_content_unchanged"])
            self.assertTrue(row["positive_ids_unchanged"])
            self.assertTrue(row["relevance_targets_unchanged"])
            self.assertTrue(row["underlying_case_unchanged"])
            self.assertTrue(row["split_unchanged"])
            self.assertTrue(row["all_non_claim_content_unchanged"])

    def test_unknown_old_claim_template_fails_closed(self) -> None:
        example = _example("case-a", "GroundLie360", "train", neutral.OCR)
        example["claim"] = 'An unregistered template says "PUBLIC NOTICE FOR case-a".'
        with self.assertRaisesRegex(NeutralBuildError, "unknown or nonconforming"):
            neutralize_example(example)


class SourceIntegrityTests(unittest.TestCase):
    def test_incomplete_source_build_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_source_fixture(Path(temporary))
            report_path = source / "build_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["status"] = "FAIL"
            report_path.write_bytes(_json_bytes(report))
            with self.assertRaisesRegex(NeutralBuildError, "not completed"):
                load_source_artifacts(source)

    def test_source_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_source_fixture(Path(temporary))
            (source / "calibration_train.sha256").write_text(
                "0" * 64 + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(NeutralBuildError, "SHA sidecar mismatch"):
                load_source_artifacts(source)

    def test_optional_manifest_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_source_fixture(Path(temporary))
            (source / "eligible_case_manifest.sha256").write_text(
                "0" * 64 + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(NeutralBuildError, "SHA sidecar mismatch"):
                load_source_artifacts(source)

    def test_malformed_source_row_fails_after_integrity_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_source_fixture(Path(temporary))
            path = source / "calibration_train.jsonl"
            path.write_bytes(b'{"calibration_example_id":\n')
            _refresh_source_hash(source, "calibration_train.jsonl")
            with self.assertRaisesRegex(NeutralBuildError, "row 0 is malformed"):
                load_source_artifacts(source)

    def test_forbidden_selector_or_veracity_field_fails_without_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_source_fixture(Path(temporary))
            path = source / "calibration_train.jsonl"
            rows = _read_jsonl(path)
            rows[0]["selection_score"] = 0.99
            path.write_bytes(_jsonl_bytes(rows))
            _refresh_source_hash(source, "calibration_train.jsonl")
            with self.assertRaisesRegex(NeutralBuildError, "forbidden scientific field"):
                load_source_artifacts(source)

    def test_formal_validation_and_test_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("Formal_Validation", "Formal Test"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        NeutralBuildError, "must not reference Formal Validation/Test"
                    ):
                        load_source_artifacts(root / name / "source")


class OverlapGateTests(unittest.TestCase):
    def test_heldout_overlap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_source_fixture(Path(temporary))
            report_path = source / "build_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["heldout_case_ids"].append("GroundLie360:case-a")
            report_path.write_bytes(_json_bytes(report))
            with self.assertRaisesRegex(NeutralBuildError, "held-out overlap"):
                load_source_artifacts(source)

    def test_train_dev_overlap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = _write_source_fixture(Path(temporary))
            path = source / "calibration_dev.jsonl"
            rows = _read_jsonl(path)
            for row in rows:
                row["canonical_underlying_case_id"] = "GroundLie360:case-a"
                row["source_dataset"] = "GroundLie360"
                row["source_case_id"] = "GroundLie360:train:case-a"
            path.write_bytes(_jsonl_bytes(rows))
            _refresh_source_hash(source, "calibration_dev.jsonl")
            with self.assertRaisesRegex(NeutralBuildError, "Train/Dev overlap"):
                load_source_artifacts(source)


class NeutralBuildTests(unittest.TestCase):
    def test_end_to_end_revision_preserves_source_and_every_non_claim_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source_fixture(root)
            output = root / "neutral-output"
            source_before = {
                path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
            }
            report = build_neutral_calibration(
                source_dir=source,
                output_dir=output,
                expected_counts=FIXTURE_COUNTS,
            )
            source_after = {
                path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
            }
            self.assertEqual(source_before, source_after)
            self.assertEqual("PASS", report["status"])
            self.assertEqual(neutral.IMPLEMENTATION_REVISION, report["implementation_revision"])
            self.assertEqual(4, report["claim_changed_count"])
            self.assertEqual(0, report["claim_unchanged_count"])
            self.assertEqual(1, report["unique_neutral_template_prefix_count"])
            self.assertEqual(
                0.0,
                report["claim_only_template_modality_accuracy_after_neutralization"],
            )
            self.assertEqual(
                "PREFIX_UNIQUE_MODALITY_LOOKUP_AMBIGUOUS_AS_UNKNOWN",
                report["claim_only_template_modality_metric"],
            )
            self.assertEqual(0, report["heldout_overlap"])
            self.assertEqual(0, report["train_dev_overlap"])
            failure_fields = [
                key for key in report if key.endswith("_invariance_failures")
            ]
            self.assertTrue(failure_fields)
            self.assertTrue(all(report[key] == 0 for key in failure_fields))
            expected_outputs = {
                "neutral_calibration_train.jsonl",
                "neutral_calibration_train.sha256",
                "neutral_calibration_dev.jsonl",
                "neutral_calibration_dev.sha256",
                "neutral_revision_manifest.json",
                "neutral_revision_manifest.sha256",
                "neutral_build_report.json",
                "dataset_card.md",
            }
            self.assertEqual(expected_outputs, {path.name for path in output.iterdir()})
            source_rows = _read_jsonl(source / "calibration_train.jsonl")
            neutral_rows = _read_jsonl(output / "neutral_calibration_train.jsonl")
            for original, revised in zip(source_rows, neutral_rows):
                original_without_claim = dict(original)
                revised_without_claim = dict(revised)
                original_without_claim.pop("claim")
                revised_without_claim.pop("claim")
                self.assertEqual(original_without_claim, revised_without_claim)
                self.assertNotEqual(original["claim"], revised["claim"])
                self.assertTrue(revised["claim"].startswith(neutral.NEUTRAL_PREFIX))
            self.assertEqual(
                sha256_file(output / "neutral_calibration_train.jsonl"),
                (output / "neutral_calibration_train.sha256").read_text(
                    encoding="utf-8"
                ).strip(),
            )
            manifest = json.loads(
                (output / "neutral_revision_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(4, manifest["example_count"])
            self.assertTrue(
                all(row["all_non_claim_content_unchanged"] for row in manifest["examples"])
            )
            required_report_fields = {
                "source_calibration_train_sha256",
                "source_calibration_dev_sha256",
                "neutral_train_sha256",
                "neutral_dev_sha256",
                "source_case_count",
                "neutral_case_count",
                "dataset_case_counts",
                "dataset_split_counts",
                "candidate_order_invariance_failures",
                "relevance_target_invariance_failures",
                "selection_outputs_inspected",
                "veracity_labels_inspected",
                "formal_validation_accessed",
                "formal_test_accessed",
                "model_loaded",
                "checkpoint_loaded",
                "training_started",
                "production_or_model_code_changed",
            }
            self.assertTrue(required_report_fields <= set(report))

    def test_count_gate_fails_without_writing_outputs(self) -> None:
        wrong_counts = ExpectedCounts(
            total_cases=3,
            total_examples=4,
            train_cases=2,
            dev_cases=1,
            ocr_examples=2,
            transcript_examples=2,
            dataset_case_counts=(("GroundLie360", 2), ("TRUE-3MFact", 1)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source_fixture(root)
            output = root / "neutral-output"
            with self.assertRaisesRegex(
                NeutralBuildError, "scientific gate failed"
            ) as raised:
                build_neutral_calibration(
                    source_dir=source,
                    output_dir=output,
                    expected_counts=wrong_counts,
                )
            self.assertIsNotNone(raised.exception.report)
            self.assertEqual("FAIL", raised.exception.report["status"])
            self.assertFalse(output.exists())

    def test_nonempty_output_directory_is_preserved_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source_fixture(root)
            output = root / "neutral-output"
            output.mkdir()
            existing = output / "existing.txt"
            existing.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(NeutralBuildError, "absent or empty"):
                build_neutral_calibration(
                    source_dir=source,
                    output_dir=output,
                    expected_counts=FIXTURE_COUNTS,
                )
            self.assertEqual("preserve", existing.read_text(encoding="utf-8"))

    def test_cli_contract_has_only_source_and_output_paths(self) -> None:
        args = build_parser().parse_args(
            ["--source-dir", "/source", "--output-dir", "/output"]
        )
        self.assertEqual(Path("/source"), args.source_dir)
        self.assertEqual(Path("/output"), args.output_dir)

    def test_builder_has_no_model_checkpoint_or_runtime_imports(self) -> None:
        source = Path(neutral.__file__).read_text(encoding="utf-8")
        import_lines = [
            line for line in source.splitlines() if line.startswith(("import ", "from "))
        ]
        for forbidden in (
            "torch",
            "transformers",
            "FrozenG1",
            "Phase3A",
            "Phase4A",
            "selector_relevance_calibration.dataset_builder",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(forbidden in line for line in import_lines))


if __name__ == "__main__":
    unittest.main()
