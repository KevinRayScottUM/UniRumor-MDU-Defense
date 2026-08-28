from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.selector_relevance_independent_audit.blinding import REVIEWER_INSTRUCTIONS
from scripts.selector_relevance_review_freeze.agreement import cohen_kappa
from scripts.selector_relevance_review_freeze.review_loader import csv_bytes
from scripts.selector_relevance_review_freeze.run_freeze_adjudication import (
    build_parser as build_adjudication_parser,
    freeze_adjudication,
)
from scripts.selector_relevance_review_freeze.run_freeze_reviews import (
    build_parser as build_review_parser,
    freeze_reviews,
)
from scripts.selector_relevance_review_freeze.schemas import (
    ADJUDICATION_COLUMNS,
    EXPECTED_UNIT_COUNT,
    FINAL_GOLD_FIELDS,
    REVIEW_COLUMNS,
    ReviewFreezeError,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_with_sidecar(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    digest = _sha(payload)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, columns, rows) -> None:
    path.write_bytes(csv_bytes(columns, rows))


class ReviewFreezeFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.cohort = root / "cohort"
        self.cohort.mkdir()
        self.review_output = root / "02_review_freeze_agreement"
        self.gold_output = root / "03_final_relevance_gold"
        self.units = []
        self.case_order = []
        self.templates = {}
        self.private_mapping = {}
        self._write_cohort()

    def _write_cohort(self) -> None:
        selected_cases = []
        requests = []
        for case_index in range(30):
            dataset = "GroundLie360" if case_index < 15 else "TRUE-3MFact"
            canonical = f"{dataset}:synthetic-{case_index:03d}"
            original = f"synthetic-{case_index:03d}"
            count = 10 if case_index < 19 else 9
            self.case_order.append(canonical)
            unit_ids = []
            for position in range(count):
                unit_id = f"u-{case_index:03d}-{position:02d}"
                unit_ids.append(unit_id)
                self.units.append(
                    {
                        "dataset": dataset,
                        "canonical_case_id": canonical,
                        "original_case_id": original,
                        "unit_id": unit_id,
                        "unit_type": "transcript" if position % 2 == 0 else "ocr",
                        "modality": "text" if position % 2 == 0 else "ocr",
                        "original_candidate_position": position,
                        "claim": f"Synthetic claim {case_index:03d}",
                        "candidate_text": f"Synthetic candidate {case_index:03d}-{position:02d}",
                    }
                )
            selected_cases.append(
                {
                    "dataset": dataset,
                    "canonical_case_id": canonical,
                    "original_case_id": original,
                    "sampling_hash": hashlib.sha256(canonical.encode()).hexdigest(),
                    "model_exposed_unit_count": count,
                    "candidate_unit_ids_in_original_order": unit_ids,
                    "candidate_unit_types_in_original_order": [
                        "transcript" if index % 2 == 0 else "ocr"
                        for index in range(count)
                    ],
                    "candidate_modalities_in_original_order": [
                        "text" if index % 2 == 0 else "ocr"
                        for index in range(count)
                    ],
                }
            )
            requests.append(
                {
                    "audit_case_id": f"audit-{case_index:03d}",
                    "dataset": dataset,
                    "canonical_case_id": canonical,
                    "original_case_id": original,
                    "claim": f"Synthetic claim {case_index:03d}",
                    "candidate_units": [],
                }
            )
        self.assert_fixture_counts()
        build_report = {
            "status": "INDEPENDENT_SCORE_BLIND_AUDIT_COHORT_BUILD_PASS",
            "implementation_revision": "step2.6r-3b1-r2-v1",
            "selected_total_count": 30,
            "selected_groundlie_count": 15,
            "selected_true3m_count": 15,
            "selected_candidate_unit_count": 289,
            "reviewer_a_row_count": 289,
            "reviewer_b_row_count": 289,
            "selection_scores_accessed": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "optimizer_created": False,
            "training_started": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_heldout_reference_content_accessed": False,
        }
        selected = {
            "status": "FROZEN",
            "implementation_revision": "step2.6r-3b1-r2-v1",
            "sampling_salt": "step2.6r-3b1-independent-audit-v1",
            "selected_cases": selected_cases,
        }
        root_values = {
            "build_report.json": _json_bytes(build_report),
            "cohort_source_lock.json": _json_bytes({"status": "PASS"}),
            "eligibility_inventory.json": _json_bytes({"selected_total_count": 30}),
            "selected_case_manifest.json": _json_bytes(selected),
            "independent_relevance_audit_requests.jsonl": b"".join(
                (json.dumps(row, sort_keys=True) + "\n").encode() for row in requests
            ),
            "independent_audit_preregistration.json": _json_bytes(
                {
                    "future_step_2_6r_3b3": {
                        "coverage_gate": {
                            "minimum_evaluable_case_count": 24,
                            "frozen_total_case_count": 30,
                            "resampling_permitted": False,
                        }
                    }
                }
            ),
        }
        for name, payload in root_values.items():
            _write_with_sidecar(self.cohort / name, payload)

        units_by_case = {}
        for unit in self.units:
            units_by_case.setdefault(unit["canonical_case_id"], []).append(unit)
        for reviewer in ("A", "B"):
            case_order = self.case_order if reviewer == "A" else list(reversed(self.case_order))
            rows = []
            mapping = []
            unit_counter = 0
            for case_number, canonical in enumerate(case_order, start=1):
                review_case_id = f"{reviewer}-C-{case_number:03d}"
                case_units = units_by_case[canonical]
                ordered = case_units if reviewer == "A" else list(reversed(case_units))
                for unit in ordered:
                    unit_counter += 1
                    review_unit_id = f"{reviewer}-U-{unit_counter:04d}"
                    rows.append(
                        {
                            "review_case_id": review_case_id,
                            "claim": unit["claim"],
                            "review_unit_id": review_unit_id,
                            "candidate_text": unit["candidate_text"],
                            "direct_relevance_label": "",
                            "review_confidence": "",
                            "review_note": "",
                        }
                    )
                    mapping.append(
                        {
                            "reviewer": reviewer,
                            "review_case_id": review_case_id,
                            "review_unit_id": review_unit_id,
                            "audit_case_id": f"audit-{self.case_order.index(canonical):03d}",
                            **{
                                key: unit[key]
                                for key in (
                                    "dataset",
                                    "canonical_case_id",
                                    "original_case_id",
                                    "unit_id",
                                    "unit_type",
                                    "modality",
                                    "original_candidate_position",
                                )
                            },
                        }
                    )
            directory = self.cohort / f"reviewer_{reviewer}"
            directory.mkdir()
            template = csv_bytes(REVIEW_COLUMNS, rows)
            readme = REVIEWER_INSTRUCTIONS.encode()
            manifest = {
                "status": "READY_FOR_INDEPENDENT_REVIEW",
                "implementation_revision": "step2.6r-3b1-r2-v1",
                "artifact_type": "blinded_direct_relevance_review_packet",
                "reviewer": reviewer,
                "case_count": 30,
                "row_count": 289,
                "public_columns": list(REVIEW_COLUMNS),
                "allowed_direct_relevance_labels": [
                    "DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE"
                ],
                "allowed_review_confidence": ["HIGH", "MEDIUM", "LOW"],
                "template_sha256": _sha(template),
                "instructions_sha256": _sha(readme),
                "review_fields_initially_blank": True,
                "dataset_blind": True,
                "modality_blind": True,
                "selector_blind": True,
                "veracity_label_blind": True,
            }
            (directory / "README_REVIEWER.md").write_bytes(readme)
            (directory / "REVIEW_MANIFEST.json").write_bytes(_json_bytes(manifest))
            (directory / "relevance_review_template.csv").write_bytes(template)
            self.templates[reviewer] = rows
            self.private_mapping[reviewer] = mapping
        private = {
            "status": "PRIVATE_FROZEN_MAPPING",
            "implementation_revision": "step2.6r-3b1-r2-v1",
            "reviewer_A": self.private_mapping["A"],
            "reviewer_B": self.private_mapping["B"],
        }
        _write_with_sidecar(
            self.cohort / "private_review_mapping.json", _json_bytes(private)
        )

    def assert_fixture_counts(self) -> None:
        if len(self.units) != EXPECTED_UNIT_COUNT or len(self.case_order) != 30:
            raise AssertionError("synthetic fixture count construction failed")

    def labels(self, direct_case_count: int = 24) -> dict[tuple[str, str], str]:
        direct_cases = set(self.case_order[:direct_case_count])
        return {
            (unit["canonical_case_id"], unit["unit_id"]): (
                "DIRECT"
                if unit["canonical_case_id"] in direct_cases
                and unit["original_candidate_position"] == 0
                else "RELATED"
            )
            for unit in self.units
        }

    def write_review_returns(
        self,
        *,
        labels_a=None,
        labels_b=None,
        confidence_a: str = "HIGH",
        confidence_b: str = "LOW",
    ) -> None:
        labels_a = labels_a or self.labels()
        labels_b = labels_b or dict(labels_a)
        for reviewer, labels, confidence in (
            ("A", labels_a, confidence_a),
            ("B", labels_b, confidence_b),
        ):
            blind_to_underlying = {
                (item["review_case_id"], item["review_unit_id"]): (
                    item["canonical_case_id"], item["unit_id"]
                )
                for item in self.private_mapping[reviewer]
            }
            rows = []
            for row in self.templates[reviewer]:
                key = blind_to_underlying[(row["review_case_id"], row["review_unit_id"])]
                rows.append(
                    {
                        **row,
                        "direct_relevance_label": labels[key],
                        "review_confidence": confidence,
                        "review_note": "Synthetic annotation note.",
                    }
                )
            completed = self.root / f"STEP26R3B2_REVIEWER_{reviewer}_completed.csv"
            _write_csv(completed, REVIEW_COLUMNS, rows)
            provenance = {
                "schema_version": 1,
                "stage": "step2.6r-3b2",
                "reviewer": reviewer,
                "review_type": "independent_score_blind_direct_relevance_annotation",
                "completed_csv_sha256": _sha(completed.read_bytes()),
                "row_count": 289,
                "case_count": 30,
                "web_search_used": False,
                "external_sources_used": False,
                "other_reviewer_output_accessed": False,
                "dataset_identity_accessed": False,
                "modality_identity_accessed": False,
                "selector_outputs_accessed": False,
                "veracity_labels_accessed": False,
                "formal_validation_accessed": False,
                "formal_test_accessed": False,
                "completed_all_rows": True,
            }
            (self.root / f"STEP26R3B2_REVIEWER_{reviewer}_provenance.json").write_bytes(
                _json_bytes(provenance)
            )

    def review_path(self, reviewer: str) -> Path:
        return self.root / f"STEP26R3B2_REVIEWER_{reviewer}_completed.csv"

    def provenance_path(self, reviewer: str) -> Path:
        return self.root / f"STEP26R3B2_REVIEWER_{reviewer}_provenance.json"

    def freeze_reviews(self, output: Path | None = None) -> Path:
        return freeze_reviews(
            cohort_dir=self.cohort,
            reviewer_a_completed=self.review_path("A"),
            reviewer_a_provenance=self.provenance_path("A"),
            reviewer_b_completed=self.review_path("B"),
            reviewer_b_provenance=self.provenance_path("B"),
            output_dir=output or self.review_output,
        )

    def write_adjudication_return(self, review_dir: Path | None = None) -> None:
        directory = review_dir or self.review_output
        template = directory / "adjudication_packet" / "adjudication_template.csv"
        rows = _read_csv(template)
        mapping = json.loads(
            (directory / "private_adjudication_mapping.json").read_text()
        )["rows"]
        blind_to_key = {
            (row["adjudication_case_id"], row["adjudication_unit_id"]): (
                row["canonical_case_id"], row["unit_id"]
            )
            for row in mapping
        }
        labels_a = self.labels()
        completed_rows = []
        for row in rows:
            key = blind_to_key[
                (row["adjudication_case_id"], row["adjudication_unit_id"])
            ]
            completed_rows.append(
                {
                    **row,
                    "final_relevance_label": labels_a[key],
                    "adjudication_confidence": "MEDIUM",
                    "adjudication_note": "Synthetic adjudication note.",
                }
            )
        completed = self.root / "STEP26R3B2_ADJUDICATION_completed.csv"
        _write_csv(completed, ADJUDICATION_COLUMNS, completed_rows)
        provenance = {
            "schema_version": 1,
            "stage": "step2.6r-3b2",
            "reviewer": "ADJUDICATOR",
            "review_type": "independent_score_blind_direct_relevance_adjudication",
            "completed_csv_sha256": _sha(completed.read_bytes()),
            "row_count": len(completed_rows),
            "case_count": len({row["adjudication_case_id"] for row in completed_rows}),
            "web_search_used": False,
            "external_sources_used": False,
            "reviewer_a_labels_accessed": False,
            "reviewer_b_labels_accessed": False,
            "dataset_identity_accessed": False,
            "modality_identity_accessed": False,
            "selector_outputs_accessed": False,
            "veracity_labels_accessed": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "completed_all_rows": True,
        }
        (self.root / "STEP26R3B2_ADJUDICATION_provenance.json").write_bytes(
            _json_bytes(provenance)
        )

    def freeze_gold(self, output: Path | None = None) -> Path:
        completed = self.root / "STEP26R3B2_ADJUDICATION_completed.csv"
        provenance = self.root / "STEP26R3B2_ADJUDICATION_provenance.json"
        return freeze_adjudication(
            cohort_dir=self.cohort,
            review_freeze_dir=self.review_output,
            adjudication_completed=completed if completed.exists() else None,
            adjudication_provenance=provenance if provenance.exists() else None,
            output_dir=output or self.gold_output,
        )


class ReviewFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture_counter = 0
        self.fixture = self.new_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def new_fixture(self) -> ReviewFreezeFixture:
        self.fixture_counter += 1
        return ReviewFreezeFixture(self.root / f"fixture-{self.fixture_counter:03d}")

    def test_end_to_end_disagreement_adjudication_and_24_case_coverage(self) -> None:
        labels_a = self.fixture.labels(24)
        labels_b = dict(labels_a)
        first = (self.fixture.case_order[0], "u-000-00")
        second = (self.fixture.case_order[1], "u-001-01")
        labels_b[first] = "RELATED"
        labels_b[second] = "IRRELEVANT"
        self.fixture.write_review_returns(labels_a=labels_a, labels_b=labels_b)
        output = self.fixture.freeze_reviews()
        expected = {
            "review_source_lock.json", "review_source_lock.sha256",
            "reviewer_A_frozen.csv", "reviewer_A_frozen.sha256",
            "reviewer_A_provenance.json", "reviewer_A_provenance.sha256",
            "reviewer_B_frozen.csv", "reviewer_B_frozen.sha256",
            "reviewer_B_provenance.json", "reviewer_B_provenance.sha256",
            "agreement_report.json", "agreement_report.sha256",
            "agreement_by_case.csv", "agreement_by_case.sha256",
            "review_resolution_pre_adjudication.jsonl",
            "review_resolution_pre_adjudication.sha256",
            "private_agreement_mapping.json", "private_agreement_mapping.sha256",
            "review_freeze_report.json", "review_freeze_report.sha256",
            "private_adjudication_mapping.json",
            "private_adjudication_mapping.sha256", "adjudication_packet",
        }
        self.assertEqual({path.name for path in output.iterdir()}, expected)
        agreement = json.loads((output / "agreement_report.json").read_text())
        self.assertEqual(agreement["exact_four_class_disagreement_count"], 2)
        self.assertEqual(agreement["binary_DIRECT_vs_nonDIRECT_disagreement_count"], 1)
        self.assertTrue(agreement["cohen_kappa_four_class_defined"])
        self.assertTrue(agreement["cohen_kappa_binary_defined"])
        packet_rows = _read_csv(
            output / "adjudication_packet" / "adjudication_template.csv"
        )
        self.assertEqual(len(packet_rows), 2)
        self.assertEqual(tuple(packet_rows[0]), ADJUDICATION_COLUMNS)
        public_packet = b"".join(
            path.read_bytes()
            for path in (output / "adjudication_packet").iterdir()
        )
        for forbidden in (
            b"reviewer_a_label", b"reviewer_b_label", b"GroundLie360",
            b"TRUE-3MFact", b"canonical_case_id", b"unit_type",
            b"selection_score", b"veracity_logits",
        ):
            self.assertNotIn(forbidden, public_packet)
        self.assertNotIn("dataset", packet_rows[0])
        self.assertNotIn("modality", packet_rows[0])
        self.assertNotIn("unit_id", packet_rows[0])
        self.assertTrue(all(row["adjudication_case_id"].startswith("ADJ-") for row in packet_rows))

        duplicate = self.fixture.root / "02-review-repeat"
        repeated = self.fixture.freeze_reviews(output=duplicate)
        self.assertEqual(
            (output / "adjudication_packet" / "adjudication_template.csv").read_bytes(),
            (repeated / "adjudication_packet" / "adjudication_template.csv").read_bytes(),
        )

        self.fixture.write_adjudication_return()
        gold = self.fixture.freeze_gold()
        expected_gold_files = {
            "final_gold_source_lock.json", "final_gold_source_lock.sha256",
            "final_relevance_gold.jsonl", "final_relevance_gold.sha256",
            "review_resolution_ledger.jsonl", "review_resolution_ledger.sha256",
            "coverage_report.json", "coverage_report.sha256",
            "final_gold_manifest.json", "final_gold_manifest.sha256",
            "final_gold_freeze_report.json", "final_gold_freeze_report.sha256",
            "adjudication_frozen.csv", "adjudication_frozen.sha256",
            "adjudication_provenance.json", "adjudication_provenance.sha256",
        }
        self.assertEqual({path.name for path in gold.iterdir()}, expected_gold_files)
        final_rows = [
            json.loads(line)
            for line in (gold / "final_relevance_gold.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(final_rows), 289)
        self.assertEqual(len({row["canonical_case_id"] for row in final_rows}), 30)
        self.assertTrue(all(set(row) == set(FINAL_GOLD_FIELDS) for row in final_rows))
        self.assertTrue(
            all(
                row["binary_direct_relevance_target"]
                == (1 if row["final_relevance_label"] == "DIRECT" else 0)
                for row in final_rows
            )
        )
        coverage = json.loads((gold / "coverage_report.json").read_text())
        self.assertEqual(coverage["evaluable_case_count"], 24)
        self.assertEqual(coverage["zero_direct_positive_case_count"], 6)
        self.assertTrue(coverage["coverage_gate_pass"])
        self.assertFalse(coverage["resampling_performed"])

    def test_zero_disagreement_path_needs_no_adjudication(self) -> None:
        self.fixture.write_review_returns()
        output = self.fixture.freeze_reviews()
        report = json.loads((output / "review_freeze_report.json").read_text())
        self.assertEqual(report["disagreement_count"], 0)
        self.assertFalse(report["adjudication_required"])
        self.assertFalse((output / "adjudication_packet").exists())
        gold = self.fixture.freeze_gold()
        self.assertFalse((gold / "adjudication_frozen.csv").exists())
        final_rows = (gold / "final_relevance_gold.jsonl").read_text().splitlines()
        self.assertEqual(len(final_rows), 289)

    def test_23_case_coverage_fails_without_resampling_or_deletion(self) -> None:
        labels = self.fixture.labels(23)
        self.fixture.write_review_returns(labels_a=labels, labels_b=dict(labels))
        self.fixture.freeze_reviews()
        gold = self.fixture.freeze_gold()
        report = json.loads((gold / "final_gold_freeze_report.json").read_text())
        coverage = json.loads((gold / "coverage_report.json").read_text())
        self.assertEqual(report["status"], "INDEPENDENT_AUDIT_RELEVANCE_COVERAGE_INSUFFICIENT")
        self.assertFalse(report["coverage_gate_pass"])
        self.assertFalse(report["resampling_performed"])
        self.assertFalse(report["step_3b3_executed"])
        self.assertEqual(coverage["evaluable_case_count"], 23)
        self.assertEqual(coverage["frozen_case_count"], 30)
        self.assertEqual(coverage["frozen_unit_count"], 289)

    def test_review_csv_fail_closed_mutations(self) -> None:
        mutations = {
            "row-count": lambda rows: rows.pop(),
            "row-order": lambda rows: rows.__setitem__(slice(0, 2), [rows[1], rows[0]]),
            "claim": lambda rows: rows[0].__setitem__("claim", "changed"),
            "candidate-text": lambda rows: rows[0].__setitem__("candidate_text", "changed"),
            "case-id": lambda rows: rows[0].__setitem__("review_case_id", "changed"),
            "unit-id": lambda rows: rows[0].__setitem__("review_unit_id", "changed"),
            "duplicate-id": lambda rows: rows[1].update({
                "review_case_id": rows[0]["review_case_id"],
                "review_unit_id": rows[0]["review_unit_id"],
            }),
            "blank-label": lambda rows: rows[0].__setitem__("direct_relevance_label", ""),
            "invalid-label": lambda rows: rows[0].__setitem__("direct_relevance_label", "RELEVANT"),
            "blank-confidence": lambda rows: rows[0].__setitem__("review_confidence", ""),
            "invalid-confidence": lambda rows: rows[0].__setitem__("review_confidence", "CERTAIN"),
            "blank-note": lambda rows: rows[0].__setitem__("review_note", ""),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                fixture = self.new_fixture()
                fixture.write_review_returns()
                rows = _read_csv(fixture.review_path("A"))
                mutation(rows)
                _write_csv(fixture.review_path("A"), REVIEW_COLUMNS, rows)
                with self.assertRaises(ReviewFreezeError):
                    fixture.freeze_reviews()

    def test_reviewer_provenance_sha_and_isolation_fail_closed(self) -> None:
        for field, value in (
            ("completed_csv_sha256", "0" * 64),
            ("web_search_used", True),
            ("other_reviewer_output_accessed", True),
            ("formal_validation_accessed", True),
            ("formal_test_accessed", True),
            ("completed_all_rows", False),
        ):
            with self.subTest(field=field):
                fixture = self.new_fixture()
                fixture.write_review_returns()
                path = fixture.provenance_path("A")
                payload = json.loads(path.read_text())
                payload[field] = value
                path.write_bytes(_json_bytes(payload))
                with self.assertRaisesRegex(ReviewFreezeError, "provenance"):
                    fixture.freeze_reviews()

    def test_source_sha_revision_and_private_mapping_order_fail_closed(self) -> None:
        fixture = self.new_fixture()
        fixture.write_review_returns()
        build_report = fixture.cohort / "build_report.json"
        build_report.write_text(build_report.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ReviewFreezeError, "SHA-256 mismatch"):
            fixture.freeze_reviews()

        fixture = self.new_fixture()
        fixture.write_review_returns()
        report = json.loads((fixture.cohort / "build_report.json").read_text())
        report["implementation_revision"] = "wrong"
        _write_with_sidecar(fixture.cohort / "build_report.json", _json_bytes(report))
        with self.assertRaisesRegex(ReviewFreezeError, "implementation_revision"):
            fixture.freeze_reviews()

        fixture = self.new_fixture()
        fixture.write_review_returns()
        rows = _read_csv(fixture.review_path("B"))
        rows[0]["direct_relevance_label"] = "INVALID"
        _write_csv(fixture.review_path("B"), REVIEW_COLUMNS, rows)
        (fixture.cohort / "private_review_mapping.json").unlink()
        (fixture.cohort / "private_review_mapping.sha256").unlink()
        with self.assertRaisesRegex(ReviewFreezeError, "Reviewer B relevance label"):
            fixture.freeze_reviews()

        fixture = self.new_fixture()
        selected_path = fixture.cohort / "selected_case_manifest.json"
        selected = json.loads(selected_path.read_text())
        selected["selected_cases"][0]["canonical_case_id"] = "GroundLie360:13025004"
        _write_with_sidecar(selected_path, _json_bytes(selected))
        fixture.write_review_returns()
        with self.assertRaisesRegex(ReviewFreezeError, "identity/accounting"):
            fixture.freeze_reviews()

    def test_private_mapping_duplicate_and_set_mismatch_rejected(self) -> None:
        for name, mutation in (
            (
                "duplicate",
                lambda payload: payload["reviewer_A"].__setitem__(
                    1, dict(payload["reviewer_A"][0])
                ),
            ),
            (
                "set-mismatch",
                lambda payload: payload["reviewer_B"][0].__setitem__(
                    "unit_id", "unknown-unit"
                ),
            ),
        ):
            with self.subTest(name=name):
                fixture = self.new_fixture()
                fixture.write_review_returns()
                path = fixture.cohort / "private_review_mapping.json"
                payload = json.loads(path.read_text())
                mutation(payload)
                _write_with_sidecar(path, _json_bytes(payload))
                with self.assertRaisesRegex(ReviewFreezeError, "mapping"):
                    fixture.freeze_reviews()

    def test_cohen_kappa_four_class_and_binary_controlled(self) -> None:
        four = cohen_kappa(
            ["DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE"],
            ["DIRECT", "RELATED", "UNREADABLE", "IRRELEVANT"],
            ("DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE"),
        )
        binary = cohen_kappa(
            ["DIRECT", "DIRECT", "NON_DIRECT", "NON_DIRECT"],
            ["DIRECT", "NON_DIRECT", "NON_DIRECT", "NON_DIRECT"],
            ("DIRECT", "NON_DIRECT"),
        )
        self.assertAlmostEqual(four, 1 / 3)
        self.assertAlmostEqual(binary, 0.5)

    def test_cohen_kappa_perfect_nondegenerate_marginals(self) -> None:
        four_labels = ["DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE"]
        binary_labels = ["DIRECT", "NON_DIRECT", "DIRECT", "NON_DIRECT"]
        self.assertEqual(
            cohen_kappa(
                four_labels,
                list(four_labels),
                ("DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE"),
            ),
            1.0,
        )
        self.assertEqual(
            cohen_kappa(
                binary_labels,
                list(binary_labels),
                ("DIRECT", "NON_DIRECT"),
            ),
            1.0,
        )

    def test_cohen_kappa_degenerate_marginals_are_undefined(self) -> None:
        self.assertIsNone(
            cohen_kappa(
                ["RELATED"] * 8,
                ["RELATED"] * 8,
                ("DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE"),
            )
        )
        for single_label in ("NON_DIRECT", "DIRECT"):
            with self.subTest(single_label=single_label):
                self.assertIsNone(
                    cohen_kappa(
                        [single_label] * 8,
                        [single_label] * 8,
                        ("DIRECT", "NON_DIRECT"),
                    )
                )

    def test_undefined_kappa_is_null_and_does_not_change_resolution_or_coverage(self) -> None:
        all_direct = {
            (unit["canonical_case_id"], unit["unit_id"]): "DIRECT"
            for unit in self.fixture.units
        }
        self.fixture.write_review_returns(
            labels_a=all_direct,
            labels_b=dict(all_direct),
        )
        review_output = self.fixture.freeze_reviews()
        report_bytes = (review_output / "agreement_report.json").read_bytes()
        report = json.loads(report_bytes)
        self.assertIsNone(report["Cohen_kappa_four_class"])
        self.assertFalse(report["cohen_kappa_four_class_defined"])
        self.assertIsNone(report["Cohen_kappa_binary"])
        self.assertFalse(report["cohen_kappa_binary_defined"])
        self.assertNotIn(b"NaN", report_bytes)
        self.assertNotIn(b"Infinity", report_bytes)

        pre_resolution = [
            json.loads(line)
            for line in (
                review_output / "review_resolution_pre_adjudication.jsonl"
            ).read_text().splitlines()
        ]
        self.assertTrue(all(row["agreement"] for row in pre_resolution))
        self.assertTrue(
            all(row["pre_adjudication_status"] == "AGREED" for row in pre_resolution)
        )

        gold_output = self.fixture.freeze_gold()
        final_rows = [
            json.loads(line)
            for line in (gold_output / "final_relevance_gold.jsonl").read_text().splitlines()
        ]
        self.assertTrue(
            all(row["resolution_source"] == "REVIEWER_AGREEMENT" for row in final_rows)
        )
        coverage = json.loads((gold_output / "coverage_report.json").read_text())
        self.assertEqual(coverage["evaluable_case_count"], 30)
        self.assertTrue(coverage["coverage_gate_pass"])

    def test_confidence_never_changes_resolution(self) -> None:
        labels = self.fixture.labels()
        self.fixture.write_review_returns(
            labels_a=labels,
            labels_b=dict(labels),
            confidence_a="LOW",
            confidence_b="HIGH",
        )
        self.fixture.freeze_reviews()
        gold = self.fixture.freeze_gold()
        final_rows = [
            json.loads(line)
            for line in (gold / "final_relevance_gold.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all(row["resolution_source"] == "REVIEWER_AGREEMENT" for row in final_rows))

    def test_adjudication_return_and_provenance_fail_closed(self) -> None:
        def prepared_fixture():
            fixture = self.new_fixture()
            labels_a = fixture.labels()
            labels_b = dict(labels_a)
            labels_b[(fixture.case_order[0], "u-000-00")] = "RELATED"
            labels_b[(fixture.case_order[1], "u-001-01")] = "IRRELEVANT"
            fixture.write_review_returns(labels_a=labels_a, labels_b=labels_b)
            fixture.freeze_reviews()
            fixture.write_adjudication_return()
            return fixture

        for name, mutation in (
            ("row-order", lambda rows: rows.reverse()),
            ("immutable", lambda rows: rows[0].__setitem__("claim", "changed")),
            ("blank-label", lambda rows: rows[0].__setitem__("final_relevance_label", "")),
            ("invalid-label", lambda rows: rows[0].__setitem__("final_relevance_label", "UNKNOWN")),
            ("blank-confidence", lambda rows: rows[0].__setitem__("adjudication_confidence", "")),
            ("invalid-confidence", lambda rows: rows[0].__setitem__("adjudication_confidence", "CERTAIN")),
        ):
            with self.subTest(name=name):
                fixture = prepared_fixture()
                path = fixture.root / "STEP26R3B2_ADJUDICATION_completed.csv"
                rows = _read_csv(path)
                mutation(rows)
                _write_csv(path, ADJUDICATION_COLUMNS, rows)
                with self.assertRaises(ReviewFreezeError):
                    fixture.freeze_gold()

        fixture = prepared_fixture()
        provenance = fixture.root / "STEP26R3B2_ADJUDICATION_provenance.json"
        payload = json.loads(provenance.read_text())
        payload["reviewer_a_labels_accessed"] = True
        provenance.write_bytes(_json_bytes(payload))
        with self.assertRaisesRegex(ReviewFreezeError, "provenance"):
            fixture.freeze_gold()

        fixture = prepared_fixture()
        provenance = fixture.root / "STEP26R3B2_ADJUDICATION_provenance.json"
        payload = json.loads(provenance.read_text())
        payload["completed_csv_sha256"] = "0" * 64
        provenance.write_bytes(_json_bytes(payload))
        with self.assertRaisesRegex(ReviewFreezeError, "provenance"):
            fixture.freeze_gold()

    def test_adjudication_cannot_change_agreed_row(self) -> None:
        labels_a = self.fixture.labels()
        labels_b = dict(labels_a)
        labels_b[(self.fixture.case_order[0], "u-000-00")] = "RELATED"
        self.fixture.write_review_returns(labels_a=labels_a, labels_b=labels_b)
        self.fixture.freeze_reviews()
        mapping_path = self.fixture.review_output / "private_adjudication_mapping.json"
        mapping = json.loads(mapping_path.read_text())
        agreed = self.fixture.units[-1]
        mapping["rows"][0]["canonical_case_id"] = agreed["canonical_case_id"]
        mapping["rows"][0]["unit_id"] = agreed["unit_id"]
        mapping_sha = _write_with_sidecar(mapping_path, _json_bytes(mapping))
        freeze_report_path = self.fixture.review_output / "review_freeze_report.json"
        freeze_report = json.loads(freeze_report_path.read_text())
        freeze_report["private_adjudication_mapping_sha256"] = mapping_sha
        _write_with_sidecar(freeze_report_path, _json_bytes(freeze_report))
        self.fixture.write_adjudication_return()
        with self.assertRaisesRegex(ReviewFreezeError, "mapping|disagreement|exactly"):
            self.fixture.freeze_gold()

    def test_safety_boundaries_cli_and_paths(self) -> None:
        review_options = {
            option
            for action in build_review_parser()._actions
            for option in action.option_strings
        }
        adjudication_options = {
            option
            for action in build_adjudication_parser()._actions
            for option in action.option_strings
        }
        for forbidden in ("--selector", "--checkpoint", "--model", "--device", "--seed"):
            self.assertNotIn(forbidden, review_options | adjudication_options)
        package = Path(__file__).parents[1] / "scripts" / "selector_relevance_review_freeze"
        source = "\n".join(path.read_text() for path in package.glob("*.py"))
        for forbidden in (
            "import torch", "from torch", "transformers", "FrozenG1Runner",
            "model.forward(", '["selection_score"]', ".selection_score",
            'get("selection_score")', "veracity_logits",
        ):
            self.assertNotIn(forbidden, source)
        for name in ("torch", "transformers", "FrozenG1Runner"):
            self.assertNotIn(name, sys.modules)
        fixture = self.new_fixture()
        fixture.write_review_returns()
        with self.assertRaisesRegex(ReviewFreezeError, "Validation/Test"):
            freeze_reviews(
                cohort_dir=fixture.cohort,
                reviewer_a_completed=fixture.root / "FormalValidation" / "a.csv",
                reviewer_a_provenance=fixture.provenance_path("A"),
                reviewer_b_completed=fixture.review_path("B"),
                reviewer_b_provenance=fixture.provenance_path("B"),
                output_dir=fixture.root / "blocked",
            )


if __name__ == "__main__":
    unittest.main()
