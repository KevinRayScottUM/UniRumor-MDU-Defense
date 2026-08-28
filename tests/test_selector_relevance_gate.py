import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.selector_relevance_gate.evaluator import (
    EvaluationError,
    compare_prediction_pair,
    run_heldout_gate,
    run_invariance_smoke,
    summarize_invariance,
    verify_approved_invariance_report,
)
from scripts.selector_relevance_gate.heldout_loader import (
    EXPECTED_HELDOUT_CASE_IDS,
    ReferenceInputError,
    calibration_overlap_count,
    load_heldout_references,
    load_phase4a_replay_requests,
    sha256_file,
)
from scripts.selector_relevance_gate.metrics import (
    HeldoutRanking,
    grouped_metrics,
    ranked_unit_ids,
    reference_metrics,
)
from scripts.selector_relevance_gate.phase4a_normalizer import (
    AUTHORITATIVE_HISTORICAL_UNIT_FIELDS,
    AUTHORITATIVE_HISTORICAL_UNIT_TYPE_MODALITY_COUNTS,
    AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256,
    CPAC_CANONICAL_CASE_ID,
    EXPECTED_HISTORICAL_CANDIDATE_UNIT_COUNT,
    EXPECTED_HISTORICAL_CASE_IDS,
    EXPECTED_STAGE_A_REQUEST_COUNT,
    IMPLEMENTATION_REVISION as NORMALIZATION_REVISION,
    NORMALIZED_UNIT_FIELDS,
    Phase4ANormalizationError,
    canonical_underlying_case_id,
    prepare_invariance_requests,
    project_historical_candidates,
    request_content_sha256,
)
from scripts.selector_relevance_gate.runtime import (
    EXPECTED_SELECTOR_SHA256,
    RuntimeIntegrationError,
    TrainingArtifacts,
    _safe_torch_load,
    _validate_state_difference,
    _validated_selector_state,
    validate_training_artifacts,
)
from scripts.selector_relevance_gate.schemas import (
    EvaluationRequest,
    EvaluationUnit,
    PredictionSnapshot,
)
from scripts.selector_relevance_training.metrics import METRIC_NAMES
from scripts.selector_relevance_training.trainer import (
    AUTHORITATIVE_CHECKPOINT_SHA256,
    IMPLEMENTATION_REVISION as TRAINING_REVISION,
    SELECTOR_ID,
)


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _write_jsonl(path: Path, rows) -> str:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return sha256_file(path)


def _unit(prefix: str, index: int) -> EvaluationUnit:
    return EvaluationUnit(
        unit_id=f"{prefix}-u{index}",
        unit_type="ocr" if index % 2 else "transcript",
        modality="ocr" if index % 2 else "text",
        text=f"audited evidence {prefix} {index}",
    )


def _request(prefix: str, *, case_id: str = "Engineering:case") -> EvaluationRequest:
    return EvaluationRequest(
        request_id=prefix,
        case_id=case_id,
        dataset=case_id.split(":", 1)[0],
        claim=f"Original challenge claim for {prefix}",
        candidate_units=tuple(_unit(prefix, index) for index in range(6)),
    )


def _historical_candidate(index: int, unit_type: str, modality: str):
    return {
        "claim_atom": f"historical claim atom {index}",
        "evidence_refs": [f"evidence-{index}"],
        "evidence_text": f"historical evidence text {index}",
        "frame_ids": [f"frame-{index}"],
        "grounding": {"method": "authoritative-schema-fixture"},
        "modality": modality,
        "phase1_source": "historical-phase1",
        "relation": "supports",
        "snippet_id": f"snippet-{index}",
        "snippet_path": f"Train/snippets/{index}.json",
        "snippet_text": f"historical snippet text {index}",
        "snippet_type": unit_type,
        "source_snippet_type": unit_type,
        "supervision": {"status": "deliberately-omitted"},
        "text": f"audited historical evidence {index}",
        "unit_id": f"historical-unit-{index}",
        "unit_type": unit_type,
    }


def _historical_phase4a_rows():
    pairs = (
        [("ocr", "ocr")] * 28
        + [("title_span", "text")] * 10
        + [("transcript", "text")] * 35
    )
    row_sizes = (10, 9, 9, 9, 9, 9, 9, 9)
    rows = []
    unit_offset = 0
    for index, canonical in enumerate(EXPECTED_HISTORICAL_CASE_IDS):
        dataset, numeric_id = canonical.rsplit(":", 1)
        row_pairs = pairs[unit_offset : unit_offset + row_sizes[index]]
        rows.append(
            {
                "case_id": f"smoke::{dataset}:train:{numeric_id}",
                "dataset": dataset,
                "claim": f"Historical claim {index}",
                "candidate_units": [
                    _historical_candidate(unit_offset + unit_index, *pair)
                    for unit_index, pair in enumerate(row_pairs)
                ],
                "source_case_id": f"{dataset}:train:{numeric_id}",
                "ground_truth_label_deliberately_omitted": True,
            }
        )
        unit_offset += row_sizes[index]
    if unit_offset != EXPECTED_HISTORICAL_CANDIDATE_UNIT_COUNT:
        raise AssertionError("authoritative fixture unit accounting is invalid")
    return rows


def _prepare_normalized_fixture(root: Path):
    from scripts.selector_relevance_gate import phase4a_normalizer

    source = root / "historical_requests.jsonl"
    source_sha = _write_jsonl(source, _historical_phase4a_rows())
    output = root / "normalized"
    with mock.patch.object(
        phase4a_normalizer,
        "AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256",
        source_sha,
    ):
        manifest = prepare_invariance_requests(
            source_artifact=source,
            source_sha256=source_sha,
            output_dir=output,
        )
    replay = output / "phase4a_invariance_requests.jsonl"
    manifest_path = output / "phase4a_invariance_request_manifest.json"
    return {
        "source": source,
        "source_sha": source_sha,
        "output": output,
        "replay": replay,
        "replay_sha": sha256_file(replay),
        "manifest": manifest_path,
        "manifest_sha": sha256_file(manifest_path),
        "manifest_payload": manifest,
    }


def _snapshot(
    unit_ids,
    scores,
    *,
    unit_logit_delta: float = 0.0,
    sample_delta: float = 0.0,
    probability_delta: float = 0.0,
    prediction: str = "real",
) -> PredictionSnapshot:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return PredictionSnapshot(
        candidate_unit_ids=tuple(unit_ids),
        selection_scores=tuple(scores),
        unit_veracity_logits=tuple(
            (0.1 + unit_logit_delta, 0.9) for _ in unit_ids
        ),
        sample_logits=(0.1 + sample_delta, 0.9),
        probabilities=(0.31 + probability_delta, 0.69),
        prediction=prediction,
        top_k_unit_ids=tuple(unit_ids[index] for index in order[:5]),
    )


def _training_artifacts(root: Path) -> TrainingArtifacts:
    selector = root / "selector_head.pt"
    selector.write_bytes(b"controlled-selector-fixture")
    return TrainingArtifacts(
        training_dir=root,
        selector_path=selector,
        selector_sha256=EXPECTED_SELECTOR_SHA256,
        training_report={},
        neutral_source_hashes={
            "neutral_calibration_train.jsonl": "a" * 64,
            "neutral_calibration_dev.jsonl": "b" * 64,
            "neutral_revision_manifest.json": "c" * 64,
        },
        calibration_case_ids=("GroundLie360:other", "TRUE-3MFact:other"),
        immutable_file_hashes={},
    )


class FakeRuntime:
    encoder_hash = "encoder-hash"
    veracity_head_hash = "veracity-hash"
    original_selection_head_hash = "original-selector"
    calibrated_selection_head_hash = "calibrated-selector"
    state_difference_names = ("selection_head.bias", "selection_head.weight")

    def __init__(self, originals, calibrated):
        self.originals = originals
        self.calibrated = calibrated
        self.calls = []
        self.immutable_checked = False

    def evaluate(self, request, *, state):
        self.calls.append((state, request.request_id, request.candidate_unit_ids))
        return (self.originals if state == "original" else self.calibrated)[
            request.request_id
        ]

    def assert_immutable(self):
        self.immutable_checked = True


class RankingMetricTests(unittest.TestCase):
    def _ranking(self, scores, positives=("b",)):
        return HeldoutRanking(
            reference_id="ref",
            case_id="Dataset:case",
            dataset="Dataset",
            reference_modality="OCR",
            candidate_unit_ids=("a", "b", "c", "d", "e", "f"),
            positive_unit_ids=tuple(positives),
            selection_scores=tuple(scores),
        )

    def test_stable_candidate_order_breaks_score_ties(self):
        self.assertEqual(
            ("a", "b", "c", "d", "e", "f"),
            ranked_unit_ids(self._ranking((1, 1, 1, 1, 1, 1))),
        )

    def test_mrr_and_binary_recall_at_1_3_5(self):
        values = reference_metrics(self._ranking((6, 3, 5, 4, 2, 1)))
        self.assertEqual(4, values["best_positive_rank"])
        self.assertEqual(0.0, values["recall_at_1"])
        self.assertEqual(0.0, values["recall_at_3"])
        self.assertEqual(1.0, values["recall_at_5"])
        self.assertEqual(0.25, values["mrr"])

    def test_ndcg_at_5_uses_binary_positive_set(self):
        values = reference_metrics(
            self._ranking((6, 5, 4, 3, 2, 1), positives=("b", "d"))
        )
        expected = (1 / math.log2(3) + 1 / math.log2(5)) / (
            1 + 1 / math.log2(3)
        )
        self.assertAlmostEqual(expected, values["ndcg_at_5"])

    def test_multi_positive_uses_best_rank(self):
        values = reference_metrics(
            self._ranking((1, 2, 3, 4, 5, 6), positives=("a", "f"))
        )
        self.assertEqual(1, values["best_positive_rank"])
        self.assertEqual(1.0, values["mrr"])

    def test_metrics_are_macro_grouped_by_case_dataset_and_modality(self):
        one = self._ranking((6, 5, 4, 3, 2, 1))
        two = HeldoutRanking(
            reference_id="ref-2",
            case_id="Other:case",
            dataset="Other",
            reference_modality="TRANSCRIPT",
            candidate_unit_ids=one.candidate_unit_ids,
            positive_unit_ids=one.positive_unit_ids,
            selection_scores=(1, 6, 5, 4, 3, 2),
        )
        metrics = grouped_metrics((one, two))
        self.assertEqual(0.5, metrics["overall"]["recall_at_1"])
        self.assertEqual(2, len(metrics["by_underlying_case"]))


class Phase4ANormalizerTests(unittest.TestCase):
    def _run(self, root: Path, rows, *, output_name="normalized"):
        from scripts.selector_relevance_gate import phase4a_normalizer

        source = root / f"{output_name}-source.jsonl"
        source_sha = _write_jsonl(source, rows)
        output = root / output_name
        with mock.patch.object(
            phase4a_normalizer,
            "AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256",
            source_sha,
        ):
            manifest = prepare_invariance_requests(
                source_artifact=source,
                source_sha256=source_sha,
                output_dir=output,
            )
        return source, source_sha, output, manifest

    def test_authoritative_historical_sha_is_exact(self):
        self.assertEqual(
            "356ee750c7b95de37e5d14b481e2f5f8fb5ae1e3805ee922d016fcb0a3ab2178",
            AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256,
        )
        self.assertEqual("step2.6r-3a0-r1-v1", NORMALIZATION_REVISION)

    def test_correct_source_projects_exactly_seven_without_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _historical_phase4a_rows()
            source, source_sha, output, manifest = self._run(root, rows)
            self.assertEqual(source_sha, sha256_file(source))
            self.assertEqual(8, manifest["source_request_count"])
            self.assertEqual(1, manifest["overlap_count"])
            self.assertEqual(1, manifest["excluded_request_count"])
            self.assertEqual(7, manifest["retained_request_count"])
            self.assertEqual(
                CPAC_CANONICAL_CASE_ID,
                manifest["excluded_requests"][0]["canonical_underlying_case_id"],
            )
            normalized_path = output / "phase4a_invariance_requests.jsonl"
            normalized_rows = [
                json.loads(line)
                for line in normalized_path.read_text(encoding="utf-8").splitlines()
            ]
            expected_rows = []
            for row in rows[1:]:
                expected_rows.append(
                    {
                        "case_id": row["case_id"],
                        "dataset": row["dataset"],
                        "claim": row["claim"],
                        "candidate_units": [
                            {
                                field: candidate[field]
                                for field in NORMALIZED_UNIT_FIELDS
                            }
                            for candidate in row["candidate_units"]
                        ],
                    }
                )
            self.assertEqual(expected_rows, normalized_rows)
            self.assertEqual(EXPECTED_STAGE_A_REQUEST_COUNT, len(normalized_rows))
            self.assertTrue(all("source_case_id" not in row for row in normalized_rows))
            self.assertTrue(
                all(
                    "ground_truth_label_deliberately_omitted" not in row
                    for row in normalized_rows
                )
            )
            self.assertTrue(
                all(
                    set(candidate) == set(NORMALIZED_UNIT_FIELDS)
                    for row in normalized_rows
                    for candidate in row["candidate_units"]
                )
            )
            self.assertTrue(
                all(value == 0 for key, value in manifest.items() if key.endswith("_changed_count"))
            )
            self.assertEqual(
                AUTHORITATIVE_HISTORICAL_UNIT_TYPE_MODALITY_COUNTS,
                manifest["historical_unit_type_modality_counts"],
            )
            self.assertEqual(73, manifest["source_candidate_unit_count"])
            for field in (
                "historical_top_level_schema_verified",
                "historical_candidate_schema_verified",
                "historical_ground_truth_omission_sentinel_present",
                "historical_ground_truth_omission_sentinel_all_true",
                "historical_unit_type_modality_pairs_verified",
                "historical_unit_metadata_projected_out",
                "ground_truth_omission_sentinel_removed",
            ):
                self.assertIs(True, manifest[field])
            self.assertEqual(8, manifest["historical_ground_truth_omission_sentinel_count"])
            self.assertEqual(17, manifest["historical_unit_field_count"])
            self.assertEqual(4, manifest["normalized_unit_field_count"])
            self.assertEqual(
                {
                    "phase4a_invariance_requests.jsonl",
                    "phase4a_invariance_requests.sha256",
                    "phase4a_invariance_request_manifest.json",
                    "phase4a_invariance_request_manifest.sha256",
                },
                {path.name for path in output.iterdir()},
            )

    def test_wrong_historical_sha_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl"
            _write_jsonl(source, _historical_phase4a_rows())
            with self.assertRaisesRegex(Phase4ANormalizationError, "authoritative"):
                prepare_invariance_requests(
                    source_artifact=source,
                    source_sha256="0" * 64,
                    output_dir=Path(temporary) / "out",
                )

    def test_source_count_not_eight_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(Phase4ANormalizationError, "exactly 8"):
                self._run(root, _historical_phase4a_rows()[:-1])

    def test_missing_cpac_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _historical_phase4a_rows()
            rows[0]["case_id"] = rows[1]["case_id"]
            rows[0]["source_case_id"] = rows[1]["source_case_id"]
            with self.assertRaisesRegex(Phase4ANormalizationError, "missing the CPAC"):
                self._run(root, rows)

    def test_multiple_overlap_and_true3mfact_overlap_are_rejected(self):
        for suffix in ("multiple", "true3mfact"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = _historical_phase4a_rows()
                rows[-1] = {
                    **rows[-1],
                    "case_id": "smoke::TRUE-3MFact:train:10145403",
                    "dataset": "TRUE-3MFact",
                    "source_case_id": "TRUE-3MFact:train:10145403",
                }
                with self.assertRaisesRegex(Phase4ANormalizationError, "TRUE-3MFact"):
                    self._run(root, rows, output_name=suffix)

    def test_retained_count_not_seven_rejected(self):
        from scripts.selector_relevance_gate import phase4a_normalizer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source_sha = _write_jsonl(source, _historical_phase4a_rows())
            with mock.patch.object(
                phase4a_normalizer,
                "AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256",
                source_sha,
            ), mock.patch.object(
                phase4a_normalizer, "EXPECTED_STAGE_A_REQUEST_COUNT", 6
            ):
                with self.assertRaisesRegex(Phase4ANormalizationError, "exactly 7"):
                    prepare_invariance_requests(
                        source_artifact=source,
                        source_sha256=source_sha,
                        output_dir=root / "out",
                    )

    def test_forbidden_label_and_prediction_fields_rejected_recursively(self):
        for field in ("label", "ground_truth_label", "prediction"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows = _historical_phase4a_rows()
                rows[0]["candidate_units"][0][field] = 0
                with self.assertRaisesRegex(Phase4ANormalizationError, "forbidden"):
                    self._run(root, rows)

    def test_omission_sentinel_is_required_and_must_be_exact_boolean_true(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = _historical_phase4a_rows()
            del rows[0]["ground_truth_label_deliberately_omitted"]
            with self.assertRaisesRegex(Phase4ANormalizationError, "source schema"):
                self._run(Path(temporary), rows, output_name="missing-sentinel")
        for index, value in enumerate((False, None, "True")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                rows = _historical_phase4a_rows()
                rows[0]["ground_truth_label_deliberately_omitted"] = value
                with self.assertRaisesRegex(
                    Phase4ANormalizationError, "must be boolean true"
                ):
                    self._run(
                        Path(temporary), rows, output_name=f"invalid-sentinel-{index}"
                    )

    def test_exact_rich_candidate_schema_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = _historical_phase4a_rows()
            del rows[0]["candidate_units"][0]["claim_atom"]
            with self.assertRaisesRegex(Phase4ANormalizationError, "invalid schema"):
                self._run(Path(temporary), rows, output_name="missing-rich-field")
        with tempfile.TemporaryDirectory() as temporary:
            rows = _historical_phase4a_rows()
            rows[0]["candidate_units"][0]["unexpected"] = "not-authoritative"
            with self.assertRaisesRegex(Phase4ANormalizationError, "invalid schema"):
                self._run(Path(temporary), rows, output_name="unexpected-rich-field")

    def test_exact_historical_unit_type_modality_pairs(self):
        for unit_type, modality in (
            ("ocr", "ocr"),
            ("title_span", "text"),
            ("transcript", "text"),
        ):
            with self.subTest(unit_type=unit_type, modality=modality):
                candidate = _historical_candidate(0, unit_type, modality)
                self.assertEqual(
                    [{field: candidate[field] for field in NORMALIZED_UNIT_FIELDS}],
                    project_historical_candidates([candidate], 0),
                )
        for unit_type, modality in (
            ("title_span", "ocr"),
            ("transcript", "ocr"),
            ("ocr", "text"),
            ("unknown", "text"),
        ):
            with self.subTest(unit_type=unit_type, modality=modality):
                candidate = _historical_candidate(0, unit_type, modality)
                with self.assertRaisesRegex(
                    Phase4ANormalizationError, "not authoritative"
                ):
                    project_historical_candidates([candidate], 0)

    def test_historical_source_requires_exactly_73_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = _historical_phase4a_rows()
            rows[-1]["candidate_units"].pop()
            with self.assertRaisesRegex(Phase4ANormalizationError, "exactly 73"):
                self._run(Path(temporary), rows)

    def test_rich_metadata_is_projected_out_without_runtime_field_changes(self):
        candidate = _historical_candidate(7, "title_span", "text")
        projected = project_historical_candidates([candidate], 0)[0]
        self.assertEqual(set(NORMALIZED_UNIT_FIELDS), set(projected))
        self.assertEqual(
            tuple(candidate[field] for field in NORMALIZED_UNIT_FIELDS),
            tuple(projected[field] for field in NORMALIZED_UNIT_FIELDS),
        )
        self.assertTrue(
            set(AUTHORITATIVE_HISTORICAL_UNIT_FIELDS) - set(projected)
        )

    def test_source_case_id_is_narrow_provenance_only(self):
        self.assertEqual(
            CPAC_CANONICAL_CASE_ID,
            canonical_underlying_case_id(
                "smoke::GroundLie360:train:13025004",
                "GroundLie360:train:13025004",
            ),
        )
        with self.assertRaisesRegex(Phase4ANormalizationError, "unsupported"):
            canonical_underlying_case_id(
                "smoke::GroundLie360:dev:13025004",
                "GroundLie360:dev:13025004",
            )

    def test_formal_validation_and_test_inputs_fail_closed(self):
        from scripts.selector_relevance_gate import phase4a_normalizer

        for name in ("Validation", "Test"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                restricted = root / name
                restricted.mkdir()
                source = restricted / "source.jsonl"
                source_sha = _write_jsonl(source, _historical_phase4a_rows())
                with mock.patch.object(
                    phase4a_normalizer,
                    "AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256",
                    source_sha,
                ):
                    with self.assertRaisesRegex(Phase4ANormalizationError, "Formal"):
                        prepare_invariance_requests(
                            source_artifact=source,
                            source_sha256=source_sha,
                            output_dir=root / "out",
                        )

    def test_formal_dataset_field_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = _historical_phase4a_rows()
            rows[0]["dataset"] = "Test"
            with self.assertRaisesRegex(Phase4ANormalizationError, "Validation/Test"):
                self._run(Path(temporary), rows)


class InputLoaderTests(unittest.TestCase):
    def _heldout_payload(self, source: Path):
        source_sha = sha256_file(source)
        references = []
        for index, case_id in enumerate(EXPECTED_HELDOUT_CASE_IDS):
            prefix = f"heldout-{index}"
            units = [_unit(prefix, unit_index) for unit_index in range(6)]
            references.append(
                {
                    "reference_id": f"reference-{index}",
                    "case_id": case_id,
                    "dataset": case_id.split(":", 1)[0],
                    "claim": f"Original challenge claim {index}",
                    "candidate_units": [dict(unit.to_dict()) for unit in units],
                    "positive_unit_ids": [units[-1].unit_id],
                    "reference_modality": "OCR",
                    "source_audit_artifact_path": str(source),
                    "source_audit_artifact_sha256": source_sha,
                    "prior_original_best_positive_rank": 6,
                    "prior_original_top5_unit_ids": [unit.unit_id for unit in units[:5]],
                    "prior_candidate_unit_ids": [unit.unit_id for unit in units],
                }
            )
        references[0]["reference_id"] = "ocr_01_direct_full_banner"
        return {
            "schema_version": 1,
            "artifact_type": "preexisting_heldout_relevance_challenge_references",
            "references": references,
        }

    def _load_normalized(self, fixture, **overrides):
        from scripts.selector_relevance_gate import heldout_loader

        arguments = {
            "expected_sha256": fixture["replay_sha"],
            "manifest_path": fixture["manifest"],
            "manifest_expected_sha256": fixture["manifest_sha"],
            **overrides,
        }
        with mock.patch.object(
            heldout_loader,
            "AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256",
            fixture["source_sha"],
        ):
            return load_phase4a_replay_requests(fixture["replay"], **arguments)

    def test_normalized_phase4a_loader_requires_exact_hash_manifest_and_seven_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _prepare_normalized_fixture(Path(temporary))
            actual, manifest_sha, source, requests = self._load_normalized(fixture)
            self.assertEqual(fixture["replay_sha"], actual)
            self.assertEqual(fixture["manifest_sha"], manifest_sha)
            self.assertEqual(fixture["source"].resolve(), source)
            self.assertEqual(EXPECTED_STAGE_A_REQUEST_COUNT, len(requests))
            self.assertEqual(
                tuple(row["case_id"] for row in _historical_phase4a_rows()[1:]),
                tuple(request.request_id for request in requests),
            )
            with self.assertRaisesRegex(ReferenceInputError, "SHA-256 mismatch"):
                self._load_normalized(fixture, expected_sha256="0" * 64)

    def test_stage_a_rejects_arbitrary_seven_rows_without_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _prepare_normalized_fixture(Path(temporary))
            with self.assertRaisesRegex(ReferenceInputError, "manifest.*missing"):
                self._load_normalized(
                    fixture,
                    manifest_path=Path(temporary) / "missing.json",
                    manifest_expected_sha256="0" * 64,
                )

    def test_stage_a_rejects_forged_rows_and_matching_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _prepare_normalized_fixture(Path(temporary))
            rows = [
                json.loads(line)
                for line in fixture["replay"].read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["claim"] = "Arbitrary replacement claim"
            fixture["replay_sha"] = _write_jsonl(fixture["replay"], rows)
            payload = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
            payload["normalized_artifact_sha256"] = fixture["replay_sha"]
            payload["retained_requests"][0]["request_content_sha256"] = (
                request_content_sha256(rows[0])
            )
            fixture["manifest_sha"] = _write_json(fixture["manifest"], payload)
            with self.assertRaisesRegex(ReferenceInputError, "historical source projection"):
                self._load_normalized(fixture)

    def test_stage_a_rejects_manifest_source_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _prepare_normalized_fixture(Path(temporary))
            payload = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
            payload["source_artifact_sha256"] = "0" * 64
            fixture["manifest_sha"] = _write_json(fixture["manifest"], payload)
            with self.assertRaisesRegex(ReferenceInputError, "source_artifact_sha256"):
                self._load_normalized(fixture)

    def test_stage_a_requires_r1_schema_and_sentinel_manifest_proof(self):
        for field in (
            "historical_top_level_schema_verified",
            "historical_candidate_schema_verified",
            "historical_ground_truth_omission_sentinel_present",
            "historical_ground_truth_omission_sentinel_all_true",
            "historical_unit_type_modality_pairs_verified",
            "historical_unit_metadata_projected_out",
            "ground_truth_omission_sentinel_removed",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = _prepare_normalized_fixture(Path(temporary))
                payload = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
                payload[field] = False
                fixture["manifest_sha"] = _write_json(fixture["manifest"], payload)
                with self.assertRaisesRegex(ReferenceInputError, field):
                    self._load_normalized(fixture)

    def test_stage_a_requires_r1_unit_count_manifest_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _prepare_normalized_fixture(Path(temporary))
            payload = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
            payload["source_candidate_unit_count"] = 72
            fixture["manifest_sha"] = _write_json(fixture["manifest"], payload)
            with self.assertRaisesRegex(ReferenceInputError, "source_candidate_unit_count"):
                self._load_normalized(fixture)

    def test_phase4a_labels_and_predictions_fail_closed(self):
        for field in ("label", "prediction"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = _prepare_normalized_fixture(Path(temporary))
                rows = [
                    json.loads(line)
                    for line in fixture["replay"].read_text(encoding="utf-8").splitlines()
                ]
                rows[0][field] = 0
                fixture["replay_sha"] = _write_jsonl(fixture["replay"], rows)
                with self.assertRaisesRegex(ReferenceInputError, "forbidden labels"):
                    self._load_normalized(fixture)

    def test_stage_a_manifest_rejects_all_six_heldout_identities(self):
        for protected in EXPECTED_HELDOUT_CASE_IDS:
            with self.subTest(protected=protected), tempfile.TemporaryDirectory() as temporary:
                fixture = _prepare_normalized_fixture(Path(temporary))
                payload = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
                payload["retained_requests"][0][
                    "canonical_underlying_case_id"
                ] = protected
                fixture["manifest_sha"] = _write_json(fixture["manifest"], payload)
                with self.assertRaisesRegex(ReferenceInputError, "protected held-out"):
                    self._load_normalized(fixture)

    def test_heldout_loader_preserves_original_claims_and_exact_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "immutable-audit.json"
            source.write_text("{}\n", encoding="utf-8")
            path = root / "references.json"
            digest = _write_json(path, self._heldout_payload(source))
            _, references = load_heldout_references(path, expected_sha256=digest)
            self.assertEqual(set(EXPECTED_HELDOUT_CASE_IDS), {item.case_id for item in references})
            self.assertTrue(all(item.claim.startswith("Original challenge") for item in references))

    def test_heldout_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audit.json"
            source.write_text("{}", encoding="utf-8")
            payload = self._heldout_payload(source)
            payload["references"][0]["case_id"] = "GroundLie360:wrong"
            path = root / "references.json"
            digest = _write_json(path, payload)
            with self.assertRaisesRegex(ReferenceInputError, "identity mismatch"):
                load_heldout_references(path, expected_sha256=digest)

    def test_neutral_synthetic_claim_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audit.json"
            source.write_text("{}", encoding="utf-8")
            payload = self._heldout_payload(source)
            payload["references"][0]["claim"] = 'The relevant content states "x".'
            path = root / "references.json"
            digest = _write_json(path, payload)
            with self.assertRaisesRegex(ReferenceInputError, "neutral synthetic"):
                load_heldout_references(path, expected_sha256=digest)

    def test_missing_positive_or_positive_outside_pool_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audit.json"
            source.write_text("{}", encoding="utf-8")
            for positives in ([], ["missing-unit"]):
                payload = self._heldout_payload(source)
                payload["references"][0]["positive_unit_ids"] = positives
                path = root / f"references-{len(positives)}.json"
                digest = _write_json(path, payload)
                with self.assertRaises((ReferenceInputError, ValueError)):
                    load_heldout_references(path, expected_sha256=digest)

    def test_source_audit_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audit.json"
            source.write_text("{}", encoding="utf-8")
            payload = self._heldout_payload(source)
            payload["references"][0]["source_audit_artifact_sha256"] = "0" * 64
            path = root / "references.json"
            digest = _write_json(path, payload)
            with self.assertRaisesRegex(ReferenceInputError, "source_audit_artifact SHA"):
                load_heldout_references(path, expected_sha256=digest)

    def test_calibration_overlap_count_is_exact(self):
        self.assertEqual(
            1,
            calibration_overlap_count(
                EXPECTED_HELDOUT_CASE_IDS,
                ("Other:case", EXPECTED_HELDOUT_CASE_IDS[2]),
            ),
        )


class InvarianceTests(unittest.TestCase):
    def setUp(self):
        self.request = _request("invariance")
        ids = self.request.candidate_unit_ids
        self.original = _snapshot(ids, (6, 5, 4, 3, 2, 1))
        self.calibrated = _snapshot(ids, (1, 2, 3, 4, 5, 6))

    def test_selection_changes_are_allowed(self):
        comparison = compare_prediction_pair(
            self.request, self.original, self.calibrated
        )
        self.assertTrue(comparison["selection_score_changed"])
        self.assertTrue(comparison["prediction_invariant"])

    def test_candidate_id_and_order_mismatches_fail(self):
        ids = list(self.request.candidate_unit_ids)
        reordered = tuple(reversed(ids))
        comparison = compare_prediction_pair(
            self.request,
            self.original,
            _snapshot(reordered, (1, 2, 3, 4, 5, 6)),
        )
        self.assertFalse(comparison["candidate_order_identical"])
        self.assertFalse(comparison["prediction_invariant"])
        changed = tuple(ids[:-1] + ["different"])
        comparison = compare_prediction_pair(
            self.request,
            self.original,
            _snapshot(changed, (1, 2, 3, 4, 5, 6)),
        )
        self.assertFalse(comparison["candidate_ids_identical"])

    def test_each_prediction_delta_above_tolerance_fails(self):
        ids = self.request.candidate_unit_ids
        variants = (
            _snapshot(ids, (1, 2, 3, 4, 5, 6), unit_logit_delta=2e-6),
            _snapshot(ids, (1, 2, 3, 4, 5, 6), sample_delta=2e-6),
            _snapshot(ids, (1, 2, 3, 4, 5, 6), probability_delta=2e-6),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertFalse(
                    compare_prediction_pair(
                        self.request, self.original, variant
                    )["prediction_invariant"]
                )

    def test_prediction_mismatch_fails(self):
        changed = _snapshot(
            self.request.candidate_unit_ids,
            (1, 2, 3, 4, 5, 6),
            prediction="fake",
        )
        self.assertFalse(
            compare_prediction_pair(
                self.request, self.original, changed
            )["prediction_invariant"]
        )

    def test_completely_unchanged_selection_fails_stage_a_gate(self):
        comparisons = (
            compare_prediction_pair(self.request, self.original, self.original),
        )
        summary = summarize_invariance(comparisons)
        self.assertTrue(summary["prediction_invariance_gate"])
        self.assertFalse(summary["selection_scores_changed"])

    def test_stage_a_writes_exact_outputs_and_does_not_touch_heldout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = tuple(
                _request(f"replay-{index}")
                for index in range(EXPECTED_STAGE_A_REQUEST_COUNT)
            )
            originals = {
                item.request_id: _snapshot(
                    item.candidate_unit_ids, (6, 5, 4, 3, 2, 1)
                )
                for item in requests
            }
            calibrated = {
                item.request_id: _snapshot(
                    item.candidate_unit_ids, (1, 2, 3, 4, 5, 6)
                )
                for item in requests
            }
            runtime = FakeRuntime(originals, calibrated)
            report = run_invariance_smoke(
                requests=requests,
                phase4a_replay_sha256="d" * 64,
                phase4a_replay_manifest_sha256="e" * 64,
                training_artifacts=_training_artifacts(root),
                runtime=runtime,
                output_dir=root / "stage-a",
            )
            self.assertEqual("PREDICTION_INVARIANCE_SMOKE_PASS", report["status"])
            self.assertFalse(report["heldout_relevance_cases_accessed"])
            self.assertFalse(report["training_started"])
            self.assertFalse(report["optimizer_created"])
            self.assertTrue(runtime.immutable_checked)
            self.assertTrue(
                (root / "stage-a/prediction_invariance_smoke_report.sha256").is_file()
            )

    def test_stage_a_rejects_wrong_request_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = FakeRuntime({}, {})
            with self.assertRaises(EvaluationError):
                run_invariance_smoke(
                    requests=(_request("only-one"),),
                    phase4a_replay_sha256="d" * 64,
                    phase4a_replay_manifest_sha256="e" * 64,
                    training_artifacts=_training_artifacts(root),
                    runtime=runtime,
                    output_dir=root / "out",
                )

    def test_stage_a_rejects_prefixed_historical_heldout_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = [
                _request(f"safe-{index}")
                for index in range(EXPECTED_STAGE_A_REQUEST_COUNT)
            ]
            requests[0] = _request(
                "protected",
                case_id="smoke::GroundLie360:train:13025004",
            )
            with self.assertRaisesRegex(EvaluationError, "held-out"):
                run_invariance_smoke(
                    requests=tuple(requests),
                    phase4a_replay_sha256="d" * 64,
                    phase4a_replay_manifest_sha256="e" * 64,
                    training_artifacts=_training_artifacts(root),
                    runtime=FakeRuntime({}, {}),
                    output_dir=root / "out",
                )


class Tensor:
    def __init__(self, shape, *, finite=True):
        self.shape = shape
        self.finite = finite

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self


class FakeTorch:
    @staticmethod
    def isfinite(tensor):
        return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: tensor.finite))


class ArtifactBoundaryTests(unittest.TestCase):
    def _payload(self, artifacts, state=None):
        return {
            "selection_head_state_dict": state
            or {"weight": Tensor((1, 4)), "bias": Tensor((1,))},
            "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
            "neutral_train_sha256": artifacts.neutral_source_hashes[
                "neutral_calibration_train.jsonl"
            ],
            "neutral_dev_sha256": artifacts.neutral_source_hashes[
                "neutral_calibration_dev.jsonl"
            ],
            "neutral_manifest_sha256": artifacts.neutral_source_hashes[
                "neutral_revision_manifest.json"
            ],
            "seed": 42,
            "selected_epoch": 1,
            "training_protocol": {},
            "optimizer_protocol": {},
            "train_class_counts": {},
            "dev_metrics": {},
            "implementation_revision": TRAINING_REVISION,
        }

    def test_selector_artifact_unexpected_tensor_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _training_artifacts(Path(temporary))
            payload = self._payload(
                artifacts,
                {
                    "weight": Tensor((1, 4)),
                    "bias": Tensor((1,)),
                    "veracity_head.weight": Tensor((2, 4)),
                },
            )
            with self.assertRaisesRegex(RuntimeIntegrationError, "only weight and bias"):
                _validated_selector_state(
                    payload,
                    original_selector_state={
                        "weight": Tensor((1, 4)),
                        "bias": Tensor((1,)),
                    },
                    torch=FakeTorch(),
                    training_artifacts=artifacts,
                )

    def test_safe_torch_loading_uses_weights_only_when_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _training_artifacts(Path(temporary))
            payload = self._payload(artifacts)

            class Torch:
                called = None

                @staticmethod
                def load(path, map_location="cpu", weights_only=False):
                    Torch.called = (path, map_location, weights_only)
                    return payload

            path = Path(temporary) / "selector.pt"
            path.write_bytes(b"fixture")
            self.assertIs(payload, _safe_torch_load(Torch, path))
            self.assertEqual((path, "cpu", True), Torch.called)

    def test_selector_shape_or_nonfinite_value_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _training_artifacts(Path(temporary))
            for state, message in (
                ({"weight": Tensor((2, 4)), "bias": Tensor((1,))}, "shape"),
                (
                    {"weight": Tensor((1, 4), finite=False), "bias": Tensor((1,))},
                    "non-finite",
                ),
            ):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(RuntimeIntegrationError, message):
                        _validated_selector_state(
                            self._payload(artifacts, state),
                            original_selector_state={
                                "weight": Tensor((1, 4)),
                                "bias": Tensor((1,)),
                            },
                            torch=FakeTorch(),
                            training_artifacts=artifacts,
                        )

    def test_selector_base_sha_or_seed_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _training_artifacts(Path(temporary))
            for field, value in (
                ("base_frozen_g1_checkpoint_sha256", "0" * 64),
                ("seed", 43),
            ):
                payload = self._payload(artifacts)
                payload[field] = value
                with self.subTest(field=field):
                    with self.assertRaisesRegex(RuntimeIntegrationError, field):
                        _validated_selector_state(
                            payload,
                            original_selector_state={
                                "weight": Tensor((1, 4)),
                                "bias": Tensor((1,)),
                            },
                            torch=FakeTorch(),
                            training_artifacts=artifacts,
                        )

    def test_nonselection_state_difference_fails_closed(self):
        original = {
            "encoder.weight": "a",
            "veracity_head.weight": "b",
            "selection_head.weight": "c",
            "selection_head.bias": "d",
        }
        with self.assertRaisesRegex(RuntimeIntegrationError, "non-selection"):
            _validate_state_difference(original, {**original, "encoder.weight": "x"})
        self.assertEqual(
            ("selection_head.weight",),
            _validate_state_difference(
                original, {**original, "selection_head.weight": "x"}
            ),
        )

    def _training_tree(self, root: Path, *, status="PASS", seed=42, base_sha=None):
        training = root / "training"
        (training / "seed_42").mkdir(parents=True)
        selector = training / "seed_42/selector_head.pt"
        selector.write_bytes(b"selector")
        summary = {
            "seeds": [42, 43, 44],
            "metrics": {
                name: {"mean": 1.0, "std": 0.0} for name in METRIC_NAMES
            },
            "future_deployment_candidate_seed": seed,
            "selection_rule": "frozen internal Dev rule",
            "candidate_dev_metrics": {},
        }
        summary_sha = _write_json(training / "multi_seed_summary.json", summary)
        neutral_hashes = {
            "neutral_calibration_train.jsonl": "a" * 64,
            "neutral_calibration_dev.jsonl": "b" * 64,
            "neutral_revision_manifest.json": "c" * 64,
        }
        report = {
            "status": status,
            "run_mode": "full",
            "implementation_revision": TRAINING_REVISION,
            "selector_id": SELECTOR_ID,
            "base_frozen_g1_checkpoint_sha256": base_sha
            or AUTHORITATIVE_CHECKPOINT_SHA256,
            "source_artifact_sha256": neutral_hashes,
            "multi_seed_summary": summary,
            "artifact_sha256": {
                "multi_seed_summary.json": summary_sha,
                "seed_42/selector_head.pt": sha256_file(selector),
            },
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "step25b_heldout_accessed": False,
            "cpac_heldout_accessed": False,
            "veracity_labels_inspected": False,
            "production_or_model_code_changed": False,
            "public_demo_changed": False,
        }
        report_path = training / "training_report.json"
        report_sha = _write_json(report_path, report)
        (training / "training_report.json.sha256").write_text(
            report_sha + "\n", encoding="utf-8"
        )
        neutral = SimpleNamespace(
            source_dir=root / "neutral",
            source_hashes=neutral_hashes,
            train_examples=(SimpleNamespace(canonical_underlying_case_id="Dataset:train"),),
            dev_examples=(SimpleNamespace(canonical_underlying_case_id="Dataset:dev"),),
        )
        return training, selector, neutral

    def test_training_report_status_seed_base_and_selector_sha_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, kwargs, expected in (
                ("status", {"status": "FAIL"}, "training report mismatch"),
                ("seed", {"seed": 43}, "seed is not 42"),
                ("base", {"base_sha": "0" * 64}, "training report mismatch"),
            ):
                case_root = root / name
                training, selector, neutral = self._training_tree(case_root, **kwargs)
                with mock.patch(
                    "scripts.selector_relevance_gate.runtime.load_neutral_data",
                    return_value=neutral,
                ), mock.patch(
                    "scripts.selector_relevance_gate.runtime.EXPECTED_SELECTOR_SHA256",
                    sha256_file(selector),
                ):
                    with self.assertRaisesRegex(RuntimeIntegrationError, expected):
                        validate_training_artifacts(training, case_root / "neutral")
            training, selector, neutral = self._training_tree(root / "selector-sha")
            with mock.patch(
                "scripts.selector_relevance_gate.runtime.load_neutral_data",
                return_value=neutral,
            ), mock.patch(
                "scripts.selector_relevance_gate.runtime.EXPECTED_SELECTOR_SHA256",
                "0" * 64,
            ):
                with self.assertRaisesRegex(RuntimeIntegrationError, "selector SHA mismatch"):
                    validate_training_artifacts(training, root / "selector-sha/neutral")


class HeldoutGateTests(unittest.TestCase):
    def _approved(self, root: Path, artifacts: TrainingArtifacts) -> Path:
        path = root / "prediction_invariance_smoke_report.json"
        report = {
            "status": "PREDICTION_INVARIANCE_SMOKE_PASS",
            "implementation_revision": "step2.6r-3-v1",
            "training_implementation_revision": TRAINING_REVISION,
            "selector_id": SELECTOR_ID,
            "deployment_candidate_seed": 42,
            "deployment_candidate_selector_sha256": artifacts.selector_sha256,
            "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
            "historical_phase4a_source_sha256": (
                AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256
            ),
            "historical_phase4a_source_request_count": 8,
            "historical_phase4a_excluded_heldout_count": 1,
            "phase4a_replay_artifact_sha256": "d" * 64,
            "phase4a_replay_manifest_sha256": "e" * 64,
            "exact_phase4a_replay_request_count": 7,
            "deterministic_nonheldout_historical_subset_used": True,
            "prediction_invariance_gate": True,
            "selection_scores_changed": True,
            "heldout_relevance_cases_accessed": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "veracity_labels_inspected": False,
            "training_started": False,
            "optimizer_created": False,
        }
        digest = _write_json(path, report)
        (root / "prediction_invariance_smoke_report.sha256").write_text(
            digest + "\n", encoding="utf-8"
        )
        return path

    def _references_and_runtime(self, root: Path, *, calibrated_scores=None, delta=0.0):
        source = root / "immutable-source-audit.json"
        source.write_text("{}\n", encoding="utf-8")
        source_sha = sha256_file(source)
        references = []
        originals = {}
        calibrated = {}
        for index, case_id in enumerate(EXPECTED_HELDOUT_CASE_IDS):
            prefix = f"heldout-{index}"
            base = _request(prefix, case_id=case_id)
            ids = base.candidate_unit_ids
            original_scores = (6, 5, 4, 3, 2, 1)
            current_scores = (
                calibrated_scores[index]
                if calibrated_scores is not None
                else (1, 2, 3, 4, 5, 10)
            )
            original = _snapshot(ids, original_scores)
            current = _snapshot(ids, current_scores, sample_delta=delta)
            references.append(
                EvaluationRequest(
                    request_id=base.request_id,
                    reference_id=(
                        "ocr_01_direct_full_banner" if index == 0 else base.request_id
                    ),
                    case_id=base.case_id,
                    dataset=base.dataset,
                    claim=base.claim,
                    candidate_units=base.candidate_units,
                    positive_unit_ids=(ids[-1],),
                    reference_modality="OCR",
                    source_audit_artifact_path=str(source),
                    source_audit_artifact_sha256=source_sha,
                    prior_original_best_positive_rank=6,
                    prior_original_top5_unit_ids=original.top_k_unit_ids,
                    prior_candidate_unit_ids=ids,
                )
            )
            originals[prefix] = original
            calibrated[prefix] = current
        return tuple(references), FakeRuntime(originals, calibrated)

    def test_stage_b_requires_approved_stage_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            missing = root / "missing.json"
            with self.assertRaisesRegex(EvaluationError, "missing"):
                verify_approved_invariance_report(missing, artifacts)

    def test_approved_stage_a_must_match_selector_and_frozen_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            path = self._approved(root, artifacts)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["deployment_candidate_selector_sha256"] = "0" * 64
            digest = _write_json(path, payload)
            path.with_suffix(".sha256").write_text(digest + "\n")
            with self.assertRaisesRegex(EvaluationError, "selector_sha256"):
                verify_approved_invariance_report(path, artifacts)

    def test_full_heldout_gate_passes_preregistered_rules_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root)
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "stage-b",
            )
            self.assertEqual(
                "HELDOUT_RELEVANCE_AND_INVARIANCE_PASS", report["status"]
            )
            self.assertTrue(report["deployment_eligible"])
            self.assertEqual(6, report["reference_rank_improvement_count"])
            self.assertTrue(report["cpac_top5_after_calibration"])
            expected = {
                "heldout_reference_manifest.json",
                "heldout_reference_manifest.sha256",
                "heldout_case_results.jsonl",
                "heldout_case_results.sha256",
                "original_selector_metrics.json",
                "calibrated_selector_metrics.json",
                "prediction_invariance_report.json",
                "heldout_relevance_gate_report.json",
                "heldout_relevance_gate_report.sha256",
                "dataset_card.md",
            }
            self.assertEqual(expected, {path.name for path in (root / "stage-b").iterdir()})

    def test_rank_regression_or_fewer_than_two_improvements_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            score_sets = [
                (1, 2, 3, 4, 5, 10),
                (6, 5, 4, 3, 2, 1),
                (6, 5, 4, 3, 2, 1),
                (6, 5, 4, 3, 2, 1),
                (6, 5, 4, 3, 2, 1),
                (6, 5, 4, 3, 2, 1),
            ]
            references, runtime = self._references_and_runtime(
                root,
                calibrated_scores=score_sets
            )
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "too-few",
            )
            self.assertEqual("HELDOUT_RELEVANCE_FAIL", report["status"])
            self.assertFalse(report["deployment_eligible"])
            self.assertEqual(1, report["reference_rank_improvement_count"])

    def test_cpac_top5_gate_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            score_sets = [(6, 5, 4, 3, 2, 1)] + [
                (1, 2, 3, 4, 5, 10)
            ] * 5
            references, runtime = self._references_and_runtime(
                root,
                calibrated_scores=score_sets
            )
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "cpac-fail",
            )
            self.assertFalse(report["cpac_top5_after_calibration"])
            self.assertFalse(report["deployment_eligible"])

    def test_prediction_invariance_failure_blocks_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root, delta=2e-6)
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "prediction-fail",
            )
            self.assertEqual("PREDICTION_INVARIANCE_FAIL", report["status"])
            self.assertFalse(report["deployment_eligible"])

    def test_positive_original_top5_cannot_be_pushed_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root)
            first = references[0]
            original = _snapshot(
                first.candidate_unit_ids, (6, 5, 4, 3, 1, 2)
            )
            calibrated = _snapshot(
                first.candidate_unit_ids, (6, 5, 4, 3, 2, 1)
            )
            runtime.originals[first.request_id] = original
            runtime.calibrated[first.request_id] = calibrated
            changed = list(references)
            changed[0] = EvaluationRequest(
                **{
                    **first.__dict__,
                    "prior_original_best_positive_rank": 5,
                    "prior_original_top5_unit_ids": original.top_k_unit_ids,
                }
            )
            report = run_heldout_gate(
                references=tuple(changed),
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "top5-regression",
            )
            rows = [
                json.loads(line)
                for line in (root / "top5-regression/heldout_case_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertFalse(rows[0]["positive_original_top5_preserved"])
            self.assertFalse(report["deployment_eligible"])

    def test_baseline_protocol_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root)
            bad = list(references)
            item = bad[0]
            bad[0] = EvaluationRequest(
                **{
                    **item.__dict__,
                    "prior_original_best_positive_rank": 5,
                }
            )
            report = run_heldout_gate(
                references=tuple(bad),
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "baseline-fail",
            )
            self.assertEqual("BASELINE_REPLAY_PROTOCOL_MISMATCH", report["status"])
            self.assertFalse(report["deployment_eligible"])

    def test_calibration_overlap_fails_before_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            artifacts = TrainingArtifacts(
                **{
                    **artifacts.__dict__,
                    "calibration_case_ids": (EXPECTED_HELDOUT_CASE_IDS[0],),
                }
            )
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root)
            with self.assertRaisesRegex(EvaluationError, "overlap"):
                run_heldout_gate(
                    references=references,
                    heldout_reference_sha256="f" * 64,
                    approved_invariance_smoke_path=approved,
                    training_artifacts=artifacts,
                    runtime=runtime,
                    output_dir=root / "overlap",
                )
            self.assertEqual([], runtime.calls)


class StaticBoundaryTests(unittest.TestCase):
    def test_evaluation_request_dummy_label_is_structural_only(self):
        request = _request("dummy")
        item = request.collator_item()
        self.assertEqual(0, item["label"])
        self.assertNotIn("label", request.__dict__)
        self.assertFalse(hasattr(PredictionSnapshot, "label"))

    def test_package_contains_no_training_or_production_dependency(self):
        package = Path(__file__).resolve().parents[1] / "scripts/selector_relevance_gate"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in package.glob("*.py")
        )
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("services.frozen_g1_runner", source)
        self.assertNotIn("webapp", source)
        self.assertNotIn("frontend", source)

    def test_cli_has_normalization_and_two_explicit_nonautomatic_modes(self):
        from scripts.selector_relevance_gate.run_gate import build_parser

        help_text = build_parser().format_help()
        self.assertIn("--prepare-invariance-requests", help_text)
        self.assertIn("--invariance-smoke", help_text)
        self.assertIn("--heldout-gate", help_text)
        self.assertIn("--approved-invariance-smoke-report", help_text)

    def test_cli_verifies_stage_a_approval_before_opening_heldout(self):
        from scripts.selector_relevance_gate import run_gate

        training = SimpleNamespace()
        with mock.patch.object(
            run_gate, "validate_training_artifacts", return_value=training
        ), mock.patch.object(
            run_gate,
            "verify_approved_invariance_report",
            side_effect=EvaluationError("approval rejected"),
        ), mock.patch.object(run_gate, "load_heldout_references") as heldout_loader:
            code = run_gate.main(
                [
                    "--heldout-gate",
                    "--project-root",
                    "/project",
                    "--phase4a-config",
                    "/config.json",
                    "--neutral-dir",
                    "/neutral",
                    "--training-dir",
                    "/training",
                    "--output-dir",
                    "/output",
                    "--approved-invariance-smoke-report",
                    "/approved.json",
                    "--heldout-reference-artifact",
                    "/heldout.json",
                    "--heldout-reference-sha256",
                    "0" * 64,
                ]
            )
        self.assertEqual(2, code)
        heldout_loader.assert_not_called()

    def test_normalization_cli_never_touches_training_or_model_runtime(self):
        from scripts.selector_relevance_gate import phase4a_normalizer, run_gate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "historical.jsonl"
            source_sha = _write_jsonl(source, _historical_phase4a_rows())
            with mock.patch.object(
                phase4a_normalizer,
                "AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256",
                source_sha,
            ), mock.patch.object(
                run_gate, "validate_training_artifacts"
            ) as training_loader, mock.patch.object(
                run_gate, "DICCEvaluationRuntime"
            ) as runtime_type:
                code = run_gate.main(
                    [
                        "--prepare-invariance-requests",
                        "--historical-phase4a-artifact",
                        str(source),
                        "--historical-phase4a-sha256",
                        source_sha,
                        "--output-dir",
                        str(root / "normalized"),
                    ]
                )
            self.assertEqual(0, code)
            training_loader.assert_not_called()
            runtime_type.assert_not_called()

    def test_stage_a_cli_requires_normalization_manifest(self):
        from scripts.selector_relevance_gate import run_gate

        with mock.patch.object(
            run_gate, "validate_training_artifacts", return_value=SimpleNamespace()
        ), mock.patch.object(run_gate, "load_phase4a_replay_requests") as loader:
            code = run_gate.main(
                [
                    "--invariance-smoke",
                    "--project-root",
                    "/project",
                    "--phase4a-config",
                    "/config.json",
                    "--neutral-dir",
                    "/neutral",
                    "--training-dir",
                    "/training",
                    "--output-dir",
                    "/output",
                    "--phase4a-replay-artifact",
                    "/normalized.jsonl",
                    "--phase4a-replay-sha256",
                    "0" * 64,
                ]
            )
        self.assertEqual(2, code)
        loader.assert_not_called()

    def test_normalizer_has_no_torch_model_checkpoint_optimizer_or_sampling_code(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/selector_relevance_gate/phase4a_normalizer.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "import torch",
            "FrozenG1Engine",
            "torch.load",
            "optimizer.step",
            ".backward(",
            "random.",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
