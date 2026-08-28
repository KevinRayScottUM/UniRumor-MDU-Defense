from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

from scripts.selector_relevance_calibration.dataset_builder import ExposureResult
from scripts.selector_fidelity_audit.cross_case import (
    canonicalize_underlying_case_id,
)
from scripts.selector_relevance_gate.phase4a_normalizer import (
    request_content_sha256,
)
from scripts.selector_relevance_independent_audit.blinding import (
    PUBLIC_PACKET_FILES,
    build_review_packet,
)
from scripts.selector_relevance_independent_audit.cohort_builder import (
    IndependentAuditBuildError,
    _safe_path,
    _summarize_exclusion_accounting,
    build_cohort,
)
from scripts.selector_relevance_independent_audit.schemas import (
    PUBLIC_REVIEW_COLUMNS,
    REVIEWER_A_SALT,
    REVIEWER_B_SALT,
    SEALED_CHALLENGE_IDS,
    STAGE_A_IDS,
    AuditCandidate,
    AuditCase,
)
from scripts.selector_relevance_independent_audit.run_build import build_parser


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return _sha(path)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return _sha(path)


class FixtureExposure:
    def normalize(self, request):
        source = request["candidate_units"]
        exposed = tuple(dict(item) for item in source[:24])
        return ExposureResult(
            candidate_units=exposed,
            source_candidate_count=len(source),
            truncated_count=max(0, len(source) - 24),
            dropped_unsupported_count=0,
        )


class VisualExposure(FixtureExposure):
    def normalize(self, request):
        result = super().normalize(request)
        values = list(result.candidate_units)
        values[0] = {**values[0], "unit_type": "visual_observation", "modality": "visual"}
        return ExposureResult(
            candidate_units=tuple(values),
            source_candidate_count=result.source_candidate_count,
            truncated_count=result.truncated_count,
            dropped_unsupported_count=result.dropped_unsupported_count,
        )


_DEFAULT_ADAPTER = object()


class IndependentAuditFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.project = root / "project"
        self.project.mkdir()
        self.neutral = root / "neutral"
        self.neutral.mkdir()
        self.output = root / "output"
        self.source = root / "g1_train.jsonl"
        self.rows = self._source_rows()
        self.expected_source_counts = self._write_source_and_lock()
        self.neutral_hashes = self._write_neutral()
        self.config = root / "phase4a_config.json"
        self.config_sha = _write_json(
            self.config,
            {
                "variant": "G1_text_ocr",
                "maximum_units_per_sample": 24,
                "max_length": 256,
            },
        )
        self._write_stage_a()

    @staticmethod
    def _candidates(prefix: str, count: int = 6) -> list[dict]:
        return [
            {
                "unit_id": f"u-{prefix}-{index}",
                "unit_type": "transcript",
                "modality": "text",
                "text": f"Candidate observation number {index}",
            }
            for index in range(count)
        ]

    def _row(self, dataset: str, case_id: str, *, count: int = 6, label: str = "fake") -> dict:
        return {
            "dataset": dataset,
            "case_id": case_id,
            "split": "train",
            "claim": "A neutral focal claim.",
            "candidate_units": self._candidates(case_id, count),
            "label": label,
            "rating": "ignored",
        }

    def _source_rows(self) -> list[dict]:
        rows = []
        for canonical in sorted(SEALED_CHALLENGE_IDS):
            dataset, case_id = canonical.split(":", 1)
            row = self._row(dataset, case_id)
            row["claim"] = "SEALED-SENTINEL-MUST-NOT-EMIT"
            row["candidate_units"][0]["text"] = "SEALED-CANDIDATE-MUST-NOT-EMIT"
            rows.append(row)
        for canonical in sorted(STAGE_A_IDS):
            dataset, case_id = canonical.split(":", 1)
            rows.append(self._row(dataset, case_id))
        rows.extend(
            [
                self._row("GroundLie360", "cal-g"),
                self._row("TRUE-3MFact", "cal-t"),
            ]
        )
        for index in range(18):
            rows.append(self._row("GroundLie360", f"g-{index:03d}", label="real"))
            rows.append(self._row("TRUE-3MFact", f"t-{index:03d}", label="fake"))
        rows.append(self._row("GroundLie360", "below-six", count=5))
        return rows

    def _write_source_and_lock(self) -> dict[str, int]:
        source_sha = _write_jsonl(self.source, self.rows)
        self.train_sha = source_sha
        self.train_lock = self.root / "phase3a_train_lock_report.json"
        self.train_lock_sha = _write_json(
            self.train_lock,
            {"status": "PASS", "source": {"path": str(self.source), "sha256": source_sha}},
        )
        counts = {"GroundLie360": 0, "TRUE-3MFact": 0}
        for row in self.rows:
            counts[row["dataset"]] += 1
        return counts

    def rewrite_source(self) -> None:
        self.expected_source_counts = self._write_source_and_lock()

    def _write_neutral(self) -> dict[str, str]:
        payloads = {
            "neutral_calibration_train.jsonl": [
                {
                    "source_dataset": "GroundLie360",
                    "source_case_id": "cal-g",
                    "canonical_underlying_case_id": "GroundLie360:cal-g",
                },
                {
                    "source_dataset": "GroundLie360",
                    "source_case_id": "13296704",
                    "canonical_underlying_case_id": "GroundLie360:13296704",
                },
                {
                    "source_dataset": "GroundLie360",
                    "source_case_id": "13025004",
                    "canonical_underlying_case_id": "GroundLie360:13025004",
                },
            ],
            "neutral_calibration_dev.jsonl": [
                {
                    "source_dataset": "TRUE-3MFact",
                    "source_case_id": "cal-t",
                    "canonical_underlying_case_id": "TRUE-3MFact:cal-t",
                }
            ],
        }
        hashes = {}
        for name, rows in payloads.items():
            value = _write_jsonl(self.neutral / name, rows)
            hashes[name] = value
            (self.neutral / (name.rsplit(".", 1)[0] + ".sha256")).write_text(
                value + "\n", encoding="utf-8"
            )
        manifest = self.neutral / "neutral_revision_manifest.json"
        manifest_hash = _write_json(manifest, {"status": "PASS", "identity_only_test_fixture": True})
        hashes[manifest.name] = manifest_hash
        (self.neutral / "neutral_revision_manifest.sha256").write_text(
            manifest_hash + "\n", encoding="utf-8"
        )
        _write_json(
            self.neutral / "neutral_build_report.json",
            {"status": "PASS", "implementation_revision": "step2.6r-1d-v1"},
        )
        return hashes

    def _write_stage_a(self) -> None:
        replay_rows = []
        retained_requests = []
        retained_row_indices = (0, 1, 3, 4, 5, 6, 7)
        for canonical, row_index in zip(sorted(STAGE_A_IDS), retained_row_indices):
            dataset, case_id = canonical.split(":", 1)
            historical_case_id = f"smoke::{dataset}:train:{case_id}"
            row = {
                "dataset": dataset,
                "case_id": historical_case_id,
                "claim": "Stage A claim",
                "candidate_units": self._candidates(f"stage-{case_id}", 9),
            }
            replay_rows.append(row)
            retained_requests.append(
                {
                    "historical_case_id": historical_case_id,
                    "source_case_id": f"{dataset}:train:{case_id}",
                    "canonical_underlying_case_id": canonical,
                    "row_index": row_index,
                    "request_content_sha256": request_content_sha256(row),
                }
            )
        self.stage_replay = self.root / "phase4a_invariance_requests.jsonl"
        self.stage_replay_sha = _write_jsonl(self.stage_replay, replay_rows)
        self.stage_manifest = self.root / "phase4a_invariance_request_manifest.json"
        self.stage_manifest_sha = _write_json(
            self.stage_manifest,
            {
                "status": "PHASE4A_INVARIANCE_REQUEST_NORMALIZATION_PASS",
                "implementation_revision": "step2.6r-3a0-r1-v1",
                "normalized_artifact_sha256": self.stage_replay_sha,
                "historical_top_level_schema_verified": True,
                "historical_candidate_schema_verified": True,
                "historical_ground_truth_omission_sentinel_all_true": True,
                "historical_unit_type_modality_pairs_verified": True,
                "historical_unit_metadata_projected_out": True,
                "source_request_count": 8,
                "source_candidate_unit_count": 73,
                "overlap_count": 1,
                "excluded_request_count": 1,
                "excluded_requests": [
                    {
                        "historical_case_id": "smoke::GroundLie360:train:13025004",
                        "source_case_id": "GroundLie360:train:13025004",
                        "canonical_underlying_case_id": "GroundLie360:13025004",
                        "row_index": 2,
                        "request_content_sha256": "0" * 64,
                        "exclusion_reason": "PREEXISTING_HELDOUT_RELEVANCE_CHALLENGE",
                    }
                ],
                "retained_request_count": 7,
                "retained_requests": retained_requests,
                "claims_changed_count": 0,
                "candidate_content_changed_count": 0,
                "candidate_text_changed_count": 0,
                "candidate_id_changed_count": 0,
                "candidate_order_changed_count": 0,
                "unit_type_changed_count": 0,
                "modality_changed_count": 0,
                "formal_validation_accessed": False,
                "formal_test_accessed": False,
                "model_loaded": False,
                "checkpoint_loaded": False,
                "selector_loaded": False,
                "training_started": False,
                "optimizer_created": False,
            },
        )
        self.stage_report = self.root / "prediction_invariance_smoke_report.json"
        report = {
            "status": "PREDICTION_INVARIANCE_SMOKE_PASS",
            "request_count": 7,
            "exact_phase4a_replay_request_count": 7,
            "historical_phase4a_source_request_count": 8,
            "historical_phase4a_excluded_heldout_count": 1,
            "candidate_id_mismatch_count": 0,
            "candidate_order_mismatch_count": 0,
            "maximum_unit_veracity_logit_difference": 0.0,
            "maximum_sample_logit_difference": 0.0,
            "maximum_probability_difference": 0.0,
            "prediction_mismatch_count": 0,
            "encoder_hash_unchanged": True,
            "veracity_head_hash_unchanged": True,
            "selection_head_hash_changed": True,
            "selection_scores_changed": True,
            "frozen_g1_checkpoint_unchanged": True,
            "prediction_invariance_gate": True,
            "deterministic_nonheldout_historical_subset_used": True,
            "heldout_relevance_cases_accessed": False,
            "veracity_labels_inspected": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "training_started": False,
            "optimizer_created": False,
            "production_or_model_code_changed": False,
            "public_demo_changed": False,
            "phase4a_replay_artifact_sha256": self.stage_replay_sha,
            "phase4a_replay_manifest_sha256": self.stage_manifest_sha,
        }
        self.stage_report_sha = _write_json(self.stage_report, report)

    def stage_replay_rows(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.stage_replay.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def stage_manifest_payload(self) -> dict:
        return json.loads(self.stage_manifest.read_text(encoding="utf-8"))

    def stage_report_payload(self) -> dict:
        return json.loads(self.stage_report.read_text(encoding="utf-8"))

    def rewrite_stage_report(self, report: dict) -> None:
        self.stage_report_sha = _write_json(self.stage_report, report)

    def rewrite_stage_manifest(self, manifest: dict, *, sync_report: bool = True) -> None:
        self.stage_manifest_sha = _write_json(self.stage_manifest, manifest)
        if sync_report:
            report = self.stage_report_payload()
            report["phase4a_replay_manifest_sha256"] = self.stage_manifest_sha
            self.rewrite_stage_report(report)

    def rewrite_stage_replay(
        self,
        rows: list[dict],
        *,
        sync_manifest: bool = True,
        sync_report: bool = True,
    ) -> None:
        self.stage_replay_sha = _write_jsonl(self.stage_replay, rows)
        if sync_manifest:
            manifest = self.stage_manifest_payload()
            manifest["normalized_artifact_sha256"] = self.stage_replay_sha
            self.rewrite_stage_manifest(manifest, sync_report=sync_report)
        elif sync_report:
            report = self.stage_report_payload()
            report["phase4a_replay_artifact_sha256"] = self.stage_replay_sha
            self.rewrite_stage_report(report)
        if sync_report and sync_manifest:
            report = self.stage_report_payload()
            report["phase4a_replay_artifact_sha256"] = self.stage_replay_sha
            report["phase4a_replay_manifest_sha256"] = self.stage_manifest_sha
            self.rewrite_stage_report(report)

    def build(
        self,
        *,
        output: Path | None = None,
        adapter=_DEFAULT_ADAPTER,
        additional_exclusion_manifests=(),
        expected_calibration_counts=None,
    ):
        selected_adapter = FixtureExposure() if adapter is _DEFAULT_ADAPTER else adapter
        return build_cohort(
            project_root=self.project,
            phase3a_train_lock_report=self.train_lock,
            phase3a_train_lock_report_sha256=self.train_lock_sha,
            phase4a_config_path=self.config,
            phase4a_config_sha256=self.config_sha,
            neutral_dir=self.neutral,
            stage_a_report_path=self.stage_report,
            stage_a_report_sha256=self.stage_report_sha,
            stage_a_replay_path=self.stage_replay,
            stage_a_replay_sha256=self.stage_replay_sha,
            stage_a_replay_manifest_path=self.stage_manifest,
            stage_a_replay_manifest_sha256=self.stage_manifest_sha,
            output_dir=output or self.output,
            additional_exclusion_manifests=additional_exclusion_manifests,
            exposure_adapter=selected_adapter,
            expected_train_sha256=self.train_sha,
            expected_source_counts=self.expected_source_counts,
            expected_neutral_hashes=self.neutral_hashes,
            expected_calibration_counts=expected_calibration_counts
            or {"GroundLie360": 3, "TRUE-3MFact": 1},
        )


class IndependentAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = IndependentAuditFixture(self.root)
        self.fixture_counter = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def fresh_fixture(self) -> IndependentAuditFixture:
        self.fixture_counter += 1
        return IndependentAuditFixture(
            self.root / f"mutation-fixture-{self.fixture_counter:03d}"
        )

    def test_build_accepts_locked_source_and_freezes_exact_balanced_cohort(self) -> None:
        result = self.fixture.build()
        report = result.build_report
        self.assertEqual(report["selected_groundlie_count"], 15)
        self.assertEqual(report["selected_true3m_count"], 15)
        self.assertEqual(report["selected_total_count"], 30)
        self.assertEqual(report["minimum_candidate_count"], 6)
        self.assertFalse(report["selection_scores_accessed"])
        self.assertFalse(report["veracity_labels_used_for_sampling"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["checkpoint_loaded"])
        self.assertFalse(report["optimizer_created"])
        self.assertFalse(report["training_started"])

    def test_authoritative_historical_stage_a_identity_uses_manifest_mapping(self) -> None:
        manifest = self.fixture.stage_manifest_payload()
        replay = self.fixture.stage_replay_rows()
        first = manifest["retained_requests"][0]
        self.assertEqual(
            replay[0]["case_id"],
            "smoke::GroundLie360:train:13199900",
        )
        self.assertEqual(first["historical_case_id"], replay[0]["case_id"])
        self.assertEqual(first["source_case_id"], "GroundLie360:train:13199900")
        self.assertEqual(
            first["canonical_underlying_case_id"], "GroundLie360:13199900"
        )
        self.assertNotEqual(
            canonicalize_underlying_case_id(
                replay[0]["dataset"], replay[0]["case_id"]
            ),
            first["canonical_underlying_case_id"],
        )
        self.fixture.build()
        self.assertEqual(
            {
                record["canonical_underlying_case_id"]
                for record in manifest["retained_requests"]
            },
            set(STAGE_A_IDS),
        )

    def test_stage_a_replay_manifest_pairing_and_uniqueness_fail_closed(self) -> None:
        mutations = (
            (
                "case-id-mismatch",
                lambda fixture: self._mutate_replay_case_id(fixture),
            ),
            (
                "simplified-local-case-id",
                lambda fixture: self._simplify_replay_case_id(fixture),
            ),
            (
                "reordered-replay",
                lambda fixture: self._reorder_replay(fixture),
            ),
            (
                "duplicate-replay",
                lambda fixture: self._duplicate_replay(fixture),
            ),
            (
                "duplicate-retained",
                lambda fixture: self._duplicate_retained(fixture),
            ),
            (
                "six-retained",
                lambda fixture: self._truncate_retained(fixture),
            ),
            (
                "six-replay",
                lambda fixture: self._truncate_replay(fixture),
            ),
            (
                "duplicate-row-index",
                lambda fixture: self._mutate_row_indices(fixture, duplicate=True),
            ),
            (
                "decreasing-row-index",
                lambda fixture: self._mutate_row_indices(fixture, duplicate=False),
            ),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                fixture = self.fresh_fixture()
                mutation(fixture)
                with self.assertRaises(IndependentAuditBuildError):
                    fixture.build()

    @staticmethod
    def _mutate_replay_case_id(fixture: IndependentAuditFixture) -> None:
        rows = fixture.stage_replay_rows()
        rows[0]["case_id"] = "smoke::GroundLie360:train:99999999"
        fixture.rewrite_stage_replay(rows)

    @staticmethod
    def _simplify_replay_case_id(fixture: IndependentAuditFixture) -> None:
        rows = fixture.stage_replay_rows()
        rows[0]["case_id"] = "13199900"
        fixture.rewrite_stage_replay(rows)

    @staticmethod
    def _reorder_replay(fixture: IndependentAuditFixture) -> None:
        rows = fixture.stage_replay_rows()
        rows[0], rows[1] = rows[1], rows[0]
        fixture.rewrite_stage_replay(rows)

    @staticmethod
    def _duplicate_replay(fixture: IndependentAuditFixture) -> None:
        rows = fixture.stage_replay_rows()
        rows[1] = dict(rows[0])
        fixture.rewrite_stage_replay(rows)

    @staticmethod
    def _duplicate_retained(fixture: IndependentAuditFixture) -> None:
        manifest = fixture.stage_manifest_payload()
        manifest["retained_requests"][1] = dict(manifest["retained_requests"][0])
        fixture.rewrite_stage_manifest(manifest)

    @staticmethod
    def _truncate_retained(fixture: IndependentAuditFixture) -> None:
        manifest = fixture.stage_manifest_payload()
        manifest["retained_requests"] = manifest["retained_requests"][:-1]
        fixture.rewrite_stage_manifest(manifest)

    @staticmethod
    def _truncate_replay(fixture: IndependentAuditFixture) -> None:
        fixture.rewrite_stage_replay(fixture.stage_replay_rows()[:-1])

    @staticmethod
    def _mutate_row_indices(
        fixture: IndependentAuditFixture, *, duplicate: bool
    ) -> None:
        manifest = fixture.stage_manifest_payload()
        if duplicate:
            manifest["retained_requests"][1]["row_index"] = manifest[
                "retained_requests"
            ][0]["row_index"]
        else:
            manifest["retained_requests"][1]["row_index"] = 9
        fixture.rewrite_stage_manifest(manifest)

    def test_stage_a_manifest_identity_and_exclusion_contract_fail_closed(self) -> None:
        def canonical(fixture, value):
            manifest = fixture.stage_manifest_payload()
            manifest["retained_requests"][0][
                "canonical_underlying_case_id"
            ] = value
            fixture.rewrite_stage_manifest(manifest)

        def excluded(fixture, *, canonical_id=None, reason=None, count=None):
            manifest = fixture.stage_manifest_payload()
            if canonical_id is not None:
                manifest["excluded_requests"][0][
                    "canonical_underlying_case_id"
                ] = canonical_id
            if reason is not None:
                manifest["excluded_requests"][0]["exclusion_reason"] = reason
            if count is not None:
                manifest["excluded_request_count"] = count
            fixture.rewrite_stage_manifest(manifest)

        mutations = (
            ("unknown-canonical", lambda f: canonical(f, "GroundLie360:99999999")),
            ("sealed-retained", lambda f: canonical(f, "GroundLie360:13025004")),
            (
                "wrong-excluded-canonical",
                lambda f: excluded(f, canonical_id="GroundLie360:13199900"),
            ),
            ("wrong-exclusion-reason", lambda f: excluded(f, reason="OTHER")),
            ("wrong-exclusion-count", lambda f: excluded(f, count=2)),
            (
                "blank-source-case-id",
                lambda f: self._mutate_retained_identity(
                    f, "source_case_id", ""
                ),
            ),
            (
                "inconsistent-source-case-id",
                lambda f: self._mutate_retained_identity(
                    f, "source_case_id", "GroundLie360:train:99999999"
                ),
            ),
            (
                "blank-historical-case-id",
                lambda f: self._mutate_retained_identity(
                    f, "historical_case_id", ""
                ),
            ),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                fixture = self.fresh_fixture()
                mutation(fixture)
                with self.assertRaises(IndependentAuditBuildError):
                    fixture.build()

    @staticmethod
    def _mutate_retained_identity(
        fixture: IndependentAuditFixture, field: str, value: str
    ) -> None:
        manifest = fixture.stage_manifest_payload()
        manifest["retained_requests"][0][field] = value
        fixture.rewrite_stage_manifest(manifest)

    def test_stage_a_manifest_scientific_contract_failures(self) -> None:
        fields_and_values = (
            ("implementation_revision", "wrong-revision"),
            ("source_request_count", 7),
            ("source_candidate_unit_count", 72),
            ("claims_changed_count", 1),
            ("candidate_content_changed_count", 1),
            ("candidate_id_changed_count", 1),
            ("candidate_order_changed_count", 1),
            ("unit_type_changed_count", 1),
            ("modality_changed_count", 1),
            ("formal_validation_accessed", True),
            ("formal_test_accessed", True),
            ("model_loaded", True),
            ("checkpoint_loaded", True),
            ("selector_loaded", True),
            ("training_started", True),
            ("optimizer_created", True),
        )
        for field, value in fields_and_values:
            with self.subTest(field=field):
                fixture = self.fresh_fixture()
                manifest = fixture.stage_manifest_payload()
                manifest[field] = value
                fixture.rewrite_stage_manifest(manifest)
                with self.assertRaises(IndependentAuditBuildError):
                    fixture.build()

    def test_stage_a_report_contract_and_sha_failures(self) -> None:
        for field, value in (
            ("request_count", 6),
            ("prediction_mismatch_count", 1),
            ("heldout_relevance_cases_accessed", True),
            ("prediction_invariance_gate", False),
            ("frozen_g1_checkpoint_unchanged", False),
        ):
            with self.subTest(field=field):
                fixture = self.fresh_fixture()
                report = fixture.stage_report_payload()
                report[field] = value
                fixture.rewrite_stage_report(report)
                with self.assertRaises(IndependentAuditBuildError):
                    fixture.build()

        for name, mutation in (
            (
                "replay-argument-sha",
                lambda fixture: setattr(fixture, "stage_replay_sha", "0" * 64),
            ),
            (
                "manifest-argument-sha",
                lambda fixture: setattr(fixture, "stage_manifest_sha", "0" * 64),
            ),
            (
                "report-replay-sha",
                lambda fixture: self._mutate_report_sha(
                    fixture, "phase4a_replay_artifact_sha256"
                ),
            ),
            (
                "report-manifest-sha",
                lambda fixture: self._mutate_report_sha(
                    fixture, "phase4a_replay_manifest_sha256"
                ),
            ),
        ):
            with self.subTest(name=name):
                fixture = self.fresh_fixture()
                mutation(fixture)
                with self.assertRaises(IndependentAuditBuildError):
                    fixture.build()

    @staticmethod
    def _mutate_report_sha(
        fixture: IndependentAuditFixture, field: str
    ) -> None:
        report = fixture.stage_report_payload()
        report[field] = "0" * 64
        fixture.rewrite_stage_report(report)

    def test_stage_a_request_content_hash_detects_claim_text_and_order_mutation(self) -> None:
        def mutate_claim(rows):
            rows[0]["claim"] = "Mutated claim"

        def mutate_text(rows):
            rows[0]["candidate_units"][0]["text"] = "Mutated candidate text"

        def mutate_order(rows):
            rows[0]["candidate_units"][0], rows[0]["candidate_units"][1] = (
                rows[0]["candidate_units"][1],
                rows[0]["candidate_units"][0],
            )

        for name, mutation in (
            ("claim", mutate_claim),
            ("candidate-text", mutate_text),
            ("candidate-order", mutate_order),
        ):
            with self.subTest(name=name):
                fixture = self.fresh_fixture()
                rows = fixture.stage_replay_rows()
                mutation(rows)
                fixture.rewrite_stage_replay(rows)
                with self.assertRaisesRegex(
                    IndependentAuditBuildError, "content SHA mismatch"
                ):
                    fixture.build()

    def test_wrong_source_sha_and_source_count_mismatches_fail_closed(self) -> None:
        original = self.fixture.train_sha
        self.fixture.train_sha = "0" * 64
        with self.assertRaisesRegex(IndependentAuditBuildError, "Train lock"):
            self.fixture.build(output=self.root / "wrong-sha")
        self.fixture.train_sha = original
        wrong_counts = dict(self.fixture.expected_source_counts)
        wrong_counts["GroundLie360"] += 1
        self.fixture.expected_source_counts = wrong_counts
        with self.assertRaisesRegex(IndependentAuditBuildError, "case count|dataset counts"):
            self.fixture.build(output=self.root / "wrong-count")

    def test_all_exclusion_categories_are_applied_and_sealed_content_never_emitted(self) -> None:
        result = self.fixture.build()
        report = result.build_report
        self.assertEqual(report["calibration_exclusion_count"], 4)
        self.assertEqual(report["sealed_challenge_exclusion_count"], 6)
        self.assertEqual(report["stage_a_exclusion_count"], 7)
        self.assertEqual(report["additional_prior_audit_exclusion_count"], 0)
        self.assertEqual(report["calibration_exclusion_membership_count"], 4)
        self.assertEqual(report["calibration_exclusion_effective_count"], 2)
        self.assertEqual(report["sealed_challenge_exclusion_effective_count"], 6)
        self.assertEqual(report["stage_a_exclusion_effective_count"], 7)
        self.assertEqual(report["calibration_stage_a_overlap_count"], 1)
        self.assertEqual(report["calibration_sealed_overlap_count"], 1)
        self.assertEqual(report["stage_a_sealed_overlap_count"], 0)
        self.assertEqual(
            report["calibration_stage_a_overlap_case_ids"],
            ["GroundLie360:13296704"],
        )
        self.assertEqual(
            report["calibration_sealed_overlap_case_ids"],
            ["GroundLie360:13025004"],
        )
        self.assertEqual(report["total_unique_excluded_case_count"], 15)
        self.assertEqual(
            report["remaining_after_identity_exclusions"],
            len(self.fixture.rows) - 15,
        )
        self.assertEqual(
            report["calibration_exclusion_membership_count_by_dataset"],
            {"GroundLie360": 3, "TRUE-3MFact": 1},
        )
        self.assertEqual(
            report["calibration_exclusion_effective_count_by_dataset"],
            {"GroundLie360": 1, "TRUE-3MFact": 1},
        )
        private_bytes = b"".join(
            path.read_bytes() for path in result.output_dir.rglob("*") if path.is_file()
        )
        self.assertNotIn(b"SEALED-SENTINEL-MUST-NOT-EMIT", private_bytes)
        self.assertNotIn(b"SEALED-CANDIDATE-MUST-NOT-EMIT", private_bytes)
        selected = json.loads((result.output_dir / "selected_case_manifest.json").read_text())
        ids = {item["canonical_case_id"] for item in selected["selected_cases"]}
        self.assertFalse(ids & SEALED_CHALLENGE_IDS)
        self.assertFalse(ids & STAGE_A_IDS)
        self.assertNotIn("GroundLie360:cal-g", ids)
        self.assertNotIn("TRUE-3MFact:cal-t", ids)

        inventory = json.loads(
            (result.output_dir / "eligibility_inventory.json").read_text()
        )
        for field in (
            "calibration_exclusion_membership_count",
            "calibration_exclusion_effective_count",
            "sealed_challenge_exclusion_membership_count",
            "sealed_challenge_exclusion_effective_count",
            "stage_a_exclusion_membership_count",
            "stage_a_exclusion_effective_count",
            "calibration_stage_a_overlap_count",
            "calibration_sealed_overlap_count",
            "stage_a_sealed_overlap_count",
            "total_unique_excluded_case_count",
            "remaining_after_identity_exclusions",
        ):
            self.assertEqual(inventory[field], report[field])
        self.assertEqual(
            sum(
                report[field]
                for field in (
                    "sealed_challenge_exclusion_effective_count",
                    "stage_a_exclusion_effective_count",
                    "calibration_exclusion_effective_count",
                    "additional_prior_audit_exclusion_effective_count",
                )
            ),
            report["total_unique_excluded_case_count"],
        )
        self.assertGreater(
            sum(
                report[field]
                for field in (
                    "sealed_challenge_exclusion_membership_count",
                    "stage_a_exclusion_membership_count",
                    "calibration_exclusion_membership_count",
                    "additional_prior_audit_exclusion_membership_count",
                )
            ),
            report["total_unique_excluded_case_count"],
        )

    def test_pairwise_overlap_precedence_and_union_are_accounted_once(self) -> None:
        source_ids = frozenset(
            {
                "GroundLie360:sealed-stage",
                "GroundLie360:stage-calibration",
                "GroundLie360:calibration-additional",
                "GroundLie360:additional-only",
            }
        )
        sealed_ids = frozenset({"GroundLie360:sealed-stage"})
        stage_a_ids = frozenset(
            {"GroundLie360:sealed-stage", "GroundLie360:stage-calibration"}
        )
        calibration_ids = frozenset(
            {
                "GroundLie360:stage-calibration",
                "GroundLie360:calibration-additional",
            }
        )
        additional_ids = frozenset(
            {
                "GroundLie360:calibration-additional",
                "GroundLie360:additional-only",
            }
        )
        report = _summarize_exclusion_accounting(
            source_ids=source_ids,
            calibration_ids=calibration_ids,
            sealed_ids=sealed_ids,
            stage_a_ids=stage_a_ids,
            additional_ids=additional_ids,
            effective_ids={
                "sealed": sealed_ids,
                "stage_a": frozenset({"GroundLie360:stage-calibration"}),
                "calibration": frozenset(
                    {"GroundLie360:calibration-additional"}
                ),
                "additional": frozenset({"GroundLie360:additional-only"}),
            },
        )
        self.assertEqual(report["stage_a_sealed_overlap_count"], 1)
        self.assertEqual(report["calibration_stage_a_overlap_count"], 1)
        self.assertEqual(report["sealed_challenge_exclusion_effective_count"], 1)
        self.assertEqual(report["stage_a_exclusion_effective_count"], 1)
        self.assertEqual(report["calibration_exclusion_effective_count"], 1)
        self.assertEqual(report["additional_prior_audit_exclusion_effective_count"], 1)
        self.assertEqual(report["additional_prior_audit_higher_priority_overlap_count"], 1)
        self.assertEqual(report["total_unique_excluded_case_count"], 4)
        self.assertEqual(report["remaining_after_exclusions"], 0)

    def test_additional_exclusions_may_overlap_higher_priority_sets(self) -> None:
        manifest = self.root / "additional_overlap.json"
        digest = _write_json(
            manifest,
            {
                "canonical_case_ids": [
                    "GroundLie360:cal-g",
                    "GroundLie360:13296704",
                    "GroundLie360:g-000",
                ]
            },
        )
        manifest.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
        result = self.fixture.build(
            additional_exclusion_manifests=(manifest,)
        )
        report = result.build_report
        self.assertEqual(report["additional_prior_audit_exclusion_membership_count"], 3)
        self.assertEqual(report["additional_prior_audit_exclusion_effective_count"], 1)
        self.assertEqual(report["additional_prior_audit_higher_priority_overlap_count"], 2)
        self.assertEqual(report["total_unique_excluded_case_count"], 16)
        self.assertEqual(report["remaining_after_exclusions"], len(self.fixture.rows) - 16)
        self.assertEqual(
            sum(
                report[field]
                for field in (
                    "sealed_challenge_exclusion_effective_count",
                    "stage_a_exclusion_effective_count",
                    "calibration_exclusion_effective_count",
                    "additional_prior_audit_exclusion_effective_count",
                )
            ),
            report["total_unique_excluded_case_count"],
        )

    def test_authoritative_identity_accounting_contract_numbers(self) -> None:
        calibration_groundlie = {
            "GroundLie360:13296704",
            "GroundLie360:13310803",
            *{f"GroundLie360:cal-{index:04d}" for index in range(568)},
        }
        calibration_true3m = {
            f"TRUE-3MFact:cal-{index:04d}" for index in range(736)
        }
        calibration_ids = frozenset(calibration_groundlie | calibration_true3m)
        sealed_ids = SEALED_CHALLENGE_IDS
        stage_a_ids = STAGE_A_IDS
        union = calibration_ids | sealed_ids | stage_a_ids
        source_ids = frozenset(
            union
            | {
                f"GroundLie360:source-{index:04d}"
                for index in range(3878 - len(union))
            }
        )
        report = _summarize_exclusion_accounting(
            source_ids=source_ids,
            calibration_ids=calibration_ids,
            sealed_ids=sealed_ids,
            stage_a_ids=stage_a_ids,
            additional_ids=frozenset(),
            effective_ids={
                "sealed": sealed_ids,
                "stage_a": stage_a_ids - sealed_ids,
                "calibration": calibration_ids - sealed_ids - stage_a_ids,
                "additional": frozenset(),
            },
        )
        self.assertEqual(report["calibration_exclusion_membership_count"], 1306)
        self.assertEqual(report["sealed_challenge_exclusion_membership_count"], 6)
        self.assertEqual(report["stage_a_exclusion_membership_count"], 7)
        self.assertEqual(report["calibration_stage_a_overlap_count"], 2)
        self.assertEqual(
            report["calibration_stage_a_overlap_case_ids"],
            ["GroundLie360:13296704", "GroundLie360:13310803"],
        )
        self.assertEqual(report["calibration_sealed_overlap_count"], 0)
        self.assertEqual(report["stage_a_sealed_overlap_count"], 0)
        self.assertEqual(report["calibration_exclusion_effective_count"], 1304)
        self.assertEqual(report["sealed_challenge_exclusion_effective_count"], 6)
        self.assertEqual(report["stage_a_exclusion_effective_count"], 7)
        self.assertEqual(report["total_unique_excluded_case_count"], 1317)
        self.assertEqual(report["remaining_after_identity_exclusions"], 2561)
        self.assertEqual(
            report["calibration_exclusion_membership_count_by_dataset"],
            {"GroundLie360": 570, "TRUE-3MFact": 736},
        )
        self.assertEqual(
            report["calibration_exclusion_effective_count_by_dataset"],
            {"GroundLie360": 568, "TRUE-3MFact": 736},
        )
        self.assertEqual(
            report["sealed_challenge_exclusion_membership_count_by_dataset"],
            {"GroundLie360": 1, "TRUE-3MFact": 5},
        )
        self.assertEqual(
            report["sealed_challenge_exclusion_effective_count_by_dataset"],
            {"GroundLie360": 1, "TRUE-3MFact": 5},
        )
        self.assertEqual(
            report["stage_a_exclusion_membership_count_by_dataset"],
            {"GroundLie360": 7, "TRUE-3MFact": 0},
        )
        self.assertEqual(
            report["stage_a_exclusion_effective_count_by_dataset"],
            {"GroundLie360": 7, "TRUE-3MFact": 0},
        )

    def test_membership_contracts_still_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            IndependentAuditBuildError, "neutral calibration identity counts mismatch"
        ):
            self.fixture.build(
                output=self.root / "wrong-calibration-membership",
                expected_calibration_counts={
                    "GroundLie360": 2,
                    "TRUE-3MFact": 1,
                },
            )
        reduced_sealed = frozenset(sorted(SEALED_CHALLENGE_IDS)[:-1])
        with patch(
            "scripts.selector_relevance_independent_audit.cohort_builder.SEALED_CHALLENGE_IDS",
            reduced_sealed,
        ):
            with self.assertRaisesRegex(
                IndependentAuditBuildError,
                "sealed challenge exclusion membership count mismatch",
            ):
                self.fixture.build(output=self.root / "wrong-sealed-membership")

    def test_formal_validation_and_test_paths_are_rejected(self) -> None:
        for name in ("Validation", "Test"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(IndependentAuditBuildError, "Validation/Test"):
                    _safe_path(self.root / name / "artifact.json", "fixture")

    def test_selection_score_fields_and_visual_exposure_are_rejected(self) -> None:
        target = next(row for row in self.fixture.rows if row["case_id"].startswith("g-"))
        target["selection_score"] = 0.5
        self.fixture.rewrite_source()
        with self.assertRaisesRegex(IndependentAuditBuildError, "forbidden"):
            self.fixture.build(output=self.root / "score")
        del target["selection_score"]
        self.fixture.rewrite_source()
        with self.assertRaisesRegex(IndependentAuditBuildError, "visual candidate"):
            self.fixture.build(output=self.root / "visual", adapter=VisualExposure())

    def test_candidate_boundaries_are_enforced(self) -> None:
        result = self.fixture.build()
        inventory = json.loads((result.output_dir / "eligibility_inventory.json").read_text())
        self.assertEqual(inventory["candidate_count_below_6_count"], 1)
        six = AuditCase(
            audit_case_id="audit-case",
            dataset="GroundLie360",
            canonical_case_id="GroundLie360:x",
            original_case_id="x",
            claim="claim",
            candidates=tuple(
                AuditCandidate(f"u{i}", "transcript", "text", "text", i)
                for i in range(6)
            ),
            sampling_hash="a" * 64,
        )
        self.assertEqual(len(six.candidates), 6)
        twenty_four = AuditCase(
            audit_case_id="audit-case-24",
            dataset="TRUE-3MFact",
            canonical_case_id="TRUE-3MFact:x",
            original_case_id="x",
            claim="claim",
            candidates=tuple(
                AuditCandidate(f"u{i}", "ocr", "ocr", "text", i)
                for i in range(24)
            ),
            sampling_hash="b" * 64,
        )
        self.assertEqual(len(twenty_four.candidates), 24)
        with self.assertRaises(ValueError):
            AuditCase(
                audit_case_id="too-many",
                dataset="GroundLie360",
                canonical_case_id="GroundLie360:y",
                original_case_id="y",
                claim="claim",
                candidates=tuple(
                    AuditCandidate(f"x{i}", "ocr", "ocr", "text", i)
                    for i in range(25)
                ),
                sampling_hash="c" * 64,
            )

    def test_sampling_is_deterministic_and_blind_to_labels_claims_and_candidate_text(self) -> None:
        first = self.fixture.build(output=self.root / "first")
        first_manifest = json.loads((first.output_dir / "selected_case_manifest.json").read_text())
        first_ids = [item["canonical_case_id"] for item in first_manifest["selected_cases"]]
        for row in self.fixture.rows:
            row["label"] = "real" if row.get("label") == "fake" else "fake"
            if row["case_id"] not in {item.split(":", 1)[1] for item in SEALED_CHALLENGE_IDS}:
                row["claim"] = "Changed but sampling-independent " + row["case_id"]
                for index, candidate in enumerate(row["candidate_units"]):
                    candidate["text"] = "Changed candidate text " + candidate["unit_id"]
                    if index % 2:
                        candidate["unit_type"] = "ocr"
                        candidate["modality"] = "ocr"
        self.fixture.rewrite_source()
        second = self.fixture.build(output=self.root / "second")
        second_manifest = json.loads((second.output_dir / "selected_case_manifest.json").read_text())
        second_ids = [item["canonical_case_id"] for item in second_manifest["selected_cases"]]
        self.assertEqual(first_ids, second_ids)

    def test_public_packets_are_independently_blinded_and_exact(self) -> None:
        result = self.fixture.build()
        packets = {}
        for reviewer in ("A", "B"):
            directory = result.output_dir / f"reviewer_{reviewer}"
            self.assertEqual(set(path.name for path in directory.iterdir()), set(PUBLIC_PACKET_FILES))
            with (directory / "relevance_review_template.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(tuple(rows[0]), PUBLIC_REVIEW_COLUMNS)
            self.assertTrue(all(not row["direct_relevance_label"] for row in rows))
            self.assertTrue(all(not row["review_confidence"] for row in rows))
            self.assertTrue(all(not row["review_note"] for row in rows))
            public_text = "\n".join((directory / name).read_text() for name in PUBLIC_PACKET_FILES)
            for forbidden in (
                "GroundLie360",
                "TRUE-3MFact",
                "canonical_case_id",
                "original_case_id",
                "unit_type",
                "selection_score",
            ):
                self.assertNotIn(forbidden, public_text)
            self.assertNotIn("modality", rows[0])
            self.assertNotIn("unit_id", rows[0])
            packets[reviewer] = rows
        self.assertEqual(len(packets["A"]), len(packets["B"]))
        self.assertNotEqual(
            [row["review_case_id"] for row in packets["A"]],
            [row["review_case_id"] for row in packets["B"]],
        )
        self.assertNotEqual(
            [row["review_unit_id"] for row in packets["A"]],
            [row["review_unit_id"] for row in packets["B"]],
        )
        mapping = json.loads(
            (result.output_dir / "private_review_mapping.json").read_text()
        )
        self.assertNotEqual(
            [item["audit_case_id"] for item in mapping["reviewer_A"]],
            [item["audit_case_id"] for item in mapping["reviewer_B"]],
        )
        public_payload = b"".join(
            (result.output_dir / f"reviewer_{reviewer}" / name).read_bytes()
            for reviewer in ("A", "B")
            for name in PUBLIC_PACKET_FILES
        )
        for overlap_identity in (
            "GroundLie360:13296704",
            "GroundLie360:13025004",
        ):
            self.assertNotIn(overlap_identity.encode(), public_payload)
        for item in mapping["reviewer_A"]:
            self.assertNotIn(item["canonical_case_id"].encode(), public_payload)
            self.assertNotIn(item["original_case_id"].encode(), public_payload)
            self.assertNotIn(item["unit_id"].encode(), public_payload)
        self.assertFalse((result.output_dir / "reviewer_A" / "private_review_mapping.json").exists())

    def test_identical_builds_are_byte_deterministic(self) -> None:
        first = self.fixture.build(output=self.root / "deterministic-a")
        second = self.fixture.build(output=self.root / "deterministic-b")
        first_files = {
            path.relative_to(first.output_dir): path.read_bytes()
            for path in first.output_dir.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(second.output_dir): path.read_bytes()
            for path in second.output_dir.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first_files, second_files)

    def test_private_mapping_restores_exact_original_candidate_order(self) -> None:
        result = self.fixture.build()
        mapping = json.loads((result.output_dir / "private_review_mapping.json").read_text())
        selected = json.loads((result.output_dir / "selected_case_manifest.json").read_text())
        expected = {
            (case["canonical_case_id"], unit_id): position
            for case in selected["selected_cases"]
            for position, unit_id in enumerate(case["candidate_unit_ids_in_original_order"])
        }
        for reviewer in ("reviewer_A", "reviewer_B"):
            observed = {
                (item["canonical_case_id"], item["unit_id"]): item["original_candidate_position"]
                for item in mapping[reviewer]
            }
            self.assertEqual(observed, expected)

    def test_preregistration_freezes_metrics_mapping_coverage_and_gate(self) -> None:
        result = self.fixture.build()
        prereg = json.loads((result.output_dir / "independent_audit_preregistration.json").read_text())
        future = prereg["future_step_2_6r_3b3"]
        self.assertEqual(future["metrics"], ["MRR", "NDCG@5", "Recall@1", "Recall@3", "Recall@5"])
        self.assertEqual(prereg["direct_relevance_binary_mapping"]["DIRECT"], 1)
        self.assertEqual(sum(prereg["direct_relevance_binary_mapping"].values()), 1)
        self.assertEqual(future["coverage_gate"]["minimum_evaluable_case_count"], 24)
        self.assertFalse(future["coverage_gate"]["resampling_permitted"])
        self.assertEqual(prereg["deployment_candidate_seed"], 42)
        self.assertTrue(prereg["prohibitions"]["seed_42_43_44_selection"])
        self.assertEqual(future["repair_verification_gate"]["minimum_absolute_mrr_or_ndcg_at_5_improvement"], 0.05)

    def test_schemas_and_imports_have_no_prediction_or_training_capability(self) -> None:
        field_names = {item.name for item in fields(AuditCase)} | {item.name for item in fields(AuditCandidate)}
        forbidden = {
            "label",
            "selection_score",
            "veracity_logits",
            "probabilities",
            "prediction",
            "checkpoint",
        }
        self.assertFalse(field_names & forbidden)
        for name in ("torch", "transformers", "optim", "FrozenG1Runner"):
            self.assertNotIn(name, sys.modules)

    def test_cli_and_builder_have_no_selector_model_checkpoint_or_training_input(self) -> None:
        options = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        forbidden_options = {
            "--selector",
            "--selector-artifact",
            "--checkpoint",
            "--model",
            "--training-dir",
            "--device",
            "--seed",
        }
        self.assertFalse(options & forbidden_options)
        source = (
            Path(__file__).parents[1]
            / "scripts"
            / "selector_relevance_independent_audit"
            / "cohort_builder.py"
        ).read_text(encoding="utf-8")
        for forbidden_import in (
            "import torch",
            "from torch",
            "FrozenG1Runner",
            "FrozenG1Engine",
            "transformers",
        ):
            self.assertNotIn(forbidden_import, source)

    def test_real_phase4a_adapter_is_required_when_no_fixture_is_injected(self) -> None:
        with self.assertRaisesRegex(IndependentAuditBuildError, "real Phase4A exposure adapter"):
            self.fixture.build(output=self.root / "no-fallback", adapter=None)


if __name__ == "__main__":
    unittest.main()
