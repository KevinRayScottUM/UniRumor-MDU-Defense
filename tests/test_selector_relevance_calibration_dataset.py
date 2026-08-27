from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from unittest.mock import patch

from scripts.selector_relevance_calibration import dataset_builder as builder
from scripts.selector_relevance_calibration.dataset_builder import (
    EXPECTED_STEP25B_HELDOUT_IDS,
    DatasetBuildError,
    ExposureResult,
    FrozenExposureUnavailableError,
    assert_no_forbidden_output_fields,
    assign_case_disjoint_splits,
    build_calibration_dataset,
    candidate_anchor_modality,
    lexical_signature,
    load_step25b_heldout_ids,
    normalize_text,
    ocr_quality_reason,
    transcript_quality_reason,
    verify_train_lock,
)
from scripts.selector_relevance_calibration.run_build_dataset import build_parser
from scripts.selector_fidelity_audit.cross_case import (
    canonicalize_underlying_case_id,
)


def _candidate(
    unit_id: str,
    unit_type: str,
    text: str,
    *,
    modality: Optional[str] = None,
    source: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    record = {
        "unit_id": unit_id,
        "unit_type": unit_type,
        "modality": modality or ("ocr" if unit_type == "ocr" else "text"),
        "text": text,
    }
    if source is not None:
        record["source"] = source
    record.update(extra)
    return record


def _balanced_candidates(prefix: str) -> list[Dict[str, Any]]:
    return [
        _candidate(
            f"{prefix}-t1",
            "transcript",
            "Experts discuss public safety during this recorded interview.",
        ),
        _candidate(
            f"{prefix}-t2",
            "transcript",
            "The speaker describes the policy decision in clear detail.",
        ),
        _candidate(
            f"{prefix}-o1",
            "ocr",
            "AMERICAN CONSERVATIVE UNION CPAC EVENT 2018",
        ),
        _candidate(
            f"{prefix}-o2",
            "ocr",
            "PUBLIC CONFERENCE SAFETY NOTICE DISPLAYED ON SCREEN",
        ),
        _candidate(
            f"{prefix}-text",
            "text",
            "Supplemental generic text remains a direct-relevance negative.",
        ),
    ]


def _row(dataset: str, case_id: str, candidates: Optional[list] = None, **extra: Any):
    record = {
        "dataset": dataset,
        "case_id": case_id,
        "claim": "The original sample claim is not used as a calibration target.",
        "candidate_units": candidates or _balanced_candidates(case_id.replace(":", "-")),
        "label": "fake",
        "veracity_label": 0,
        "selection_score": 999.0,
        "veracity_logits": {"fake": 99.0, "real": -99.0},
        "sample_logits": {"fake": 99.0, "real": -99.0},
        "probabilities": {"fake": 1.0, "real": 0.0},
        "prediction": "fake",
        "top_k_selection_units": [],
    }
    record.update(extra)
    return record


def _write_phase4a_layout(
    project: Path,
    *,
    engine_source: Optional[str] = None,
) -> Dict[str, Path]:
    phase3_dir = project / "MDU" / "scripts" / "clip12_phase3_common"
    phase4a_dir = (
        project / "MDU" / "scripts" / "clip12_phase4a_inference_handoff"
    )
    phase3_dir.mkdir(parents=True)
    phase4a_dir.mkdir(parents=True)
    files = {
        "clip12p3_common.py": phase3_dir / "clip12p3_common.py",
        "clip12p3_model.py": phase3_dir / "clip12p3_model.py",
        "clip12p4a_common.py": phase4a_dir / "clip12p4a_common.py",
        "clip12p4a_engine.py": phase4a_dir / "clip12p4a_engine.py",
    }
    files["clip12p3_common.py"].write_text(
        "def common_marker():\n"
        "    return 'phase3-common'\n",
        encoding="utf-8",
    )
    files["clip12p3_model.py"].write_text(
        "from clip12p3_common import common_marker\n\n"
        "def model_marker():\n"
        "    return common_marker() + ':model'\n",
        encoding="utf-8",
    )
    files["clip12p4a_common.py"].write_text(
        "def phase4a_marker():\n"
        "    return 'phase4a-common'\n",
        encoding="utf-8",
    )
    files["clip12p4a_engine.py"].write_text(
        engine_source
        or (
            "import sys\n"
            "from clip12p3_common import common_marker\n"
            "from clip12p3_model import model_marker\n"
            "from clip12p4a_common import phase4a_marker\n\n"
            "IMPORT_PATH_SNAPSHOT = tuple(sys.path)\n\n"
            "def normalize_request(request, config):\n"
            "    assert common_marker() == 'phase3-common'\n"
            "    assert model_marker() == 'phase3-common:model'\n"
            "    assert phase4a_marker() == 'phase4a-common'\n"
            "    limit = int(config['maximum_units_per_sample'])\n"
            "    result = dict(request)\n"
            "    result['candidate_units'] = list(request['candidate_units'])[:limit]\n"
            "    result['truncated_unit_count'] = max(0, len(request['candidate_units']) - limit)\n"
            "    result['dropped_unsupported_count'] = 0\n"
            "    return result\n"
        ),
        encoding="utf-8",
    )
    return files


class ControlledExposureAdapter:
    def __init__(self, transform=None):
        self.transform = transform or (lambda request, candidates: candidates)
        self.calls = []

    def normalize(self, request: Mapping[str, Any]) -> ExposureResult:
        self.calls.append(request)
        source = [dict(item) for item in request["candidate_units"]]
        exposed = list(self.transform(request, source))
        removed = len(source) - len(exposed)
        return ExposureResult(
            candidate_units=tuple(exposed),
            source_candidate_count=len(source),
            truncated_count=removed,
            dropped_unsupported_count=0,
        )


class GuardedMapping(Mapping):
    def __init__(self, payload: Mapping[str, Any], forbidden: Iterable[str]):
        self._payload = dict(payload)
        self._forbidden = set(forbidden)

    def __getitem__(self, key):
        if key in self._forbidden:
            raise AssertionError(f"forbidden field accessed: {key}")
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self):
        return len(self._payload)

    def get(self, key, default=None):
        if key in self._forbidden:
            raise AssertionError(f"forbidden field accessed: {key}")
        return self._payload.get(key, default)


class BuilderFixture:
    def __init__(self, root: Path, rows: list[Mapping[str, Any]]):
        self.root = root
        self.project = root / "project"
        self.project.mkdir(parents=True)
        source_root = root / "locked-source"
        source_root.mkdir(parents=True)
        self.source = source_root / "locked_train.jsonl"
        self.source.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.source_sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.lock = root / "train_lock.json"
        self.lock.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "source": {"path": str(self.source), "sha256": self.source_sha},
                }
            ),
            encoding="utf-8",
        )
        self.config = root / "phase4a_config.json"
        self.config.write_text('{"maximum_units_per_sample":24}\n', encoding="utf-8")
        self.manifest = root / "step25b.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "selected_cases": [
                        {"canonical_underlying_case_id": identity}
                        for identity in sorted(EXPECTED_STEP25B_HELDOUT_IDS)
                    ]
                }
            ),
            encoding="utf-8",
        )

    def build(self, output: Path, adapter=None):
        return build_calibration_dataset(
            project_root=self.project,
            phase3a_train_lock_report=self.lock,
            phase4a_config_path=self.config,
            step25b_selected_manifest=self.manifest,
            heldout_cases=("GroundLie360:13025004",),
            output_dir=output,
            exposure_adapter=adapter or ControlledExposureAdapter(),
            expected_train_sha256=self.source_sha,
        )


def _groundlie_row(
    case_id: str,
    *,
    split: str = "train",
    unit_case_id: Optional[str] = None,
) -> Dict[str, Any]:
    unit_case_id = unit_case_id or case_id
    candidates = _balanced_candidates(f"groundlie-{case_id}")
    candidates[0]["unit_id"] = (
        f"GroundLie360:test:{unit_case_id}:transcript:0"
    )
    candidates[2]["unit_id"] = f"GroundLie360:test:{unit_case_id}:ocr:0"
    return _row(
        "GroundLie360",
        f"GroundLie360:train:{case_id}",
        candidates,
        split=split,
    )


def _build_with_authoritative_train_lock(
    fixture: BuilderFixture,
    output: Path,
    adapter=None,
):
    train_lock = builder.TrainLock(
        source_path=fixture.source.resolve(),
        source_sha256=builder.AUTHORITATIVE_TRAIN_SHA256,
    )
    with patch.object(builder, "verify_train_lock", return_value=train_lock):
        return fixture.build(output, adapter)


class SelectorRelevanceCalibrationDatasetTests(unittest.TestCase):
    def test_cli_contract_exposes_required_paths_and_repeatable_heldout_cases(self):
        args = build_parser().parse_args(
            [
                "--project-root",
                "/project",
                "--phase3a-train-lock-report",
                "/train-lock.json",
                "--phase4a-config",
                "/phase4a.json",
                "--step25b-selected-manifest",
                "/step25b.json",
                "--heldout-case",
                "GroundLie360:13025004",
                "--heldout-case",
                "Dataset:second",
                "--output-dir",
                "/output",
            ]
        )
        self.assertEqual(Path("/project"), args.project_root)
        self.assertEqual(Path("/train-lock.json"), args.phase3a_train_lock_report)
        self.assertEqual(Path("/phase4a.json"), args.phase4a_config)
        self.assertEqual(Path("/step25b.json"), args.step25b_selected_manifest)
        self.assertEqual(
            ["GroundLie360:13025004", "Dataset:second"], args.heldout_case
        )
        self.assertEqual(Path("/output"), args.output_dir)

    def test_train_lock_requires_pass_existing_source_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = BuilderFixture(root, [_row("DatasetA", "DatasetA:train:1")])
            verified = verify_train_lock(
                fixture.project, fixture.lock, fixture.source_sha
            )
            self.assertEqual(fixture.source.resolve(), verified.source_path)

            payload = json.loads(fixture.lock.read_text())
            payload["status"] = "FAIL"
            fixture.lock.write_text(json.dumps(payload))
            with self.assertRaisesRegex(DatasetBuildError, "status must be PASS"):
                verify_train_lock(fixture.project, fixture.lock, fixture.source_sha)

    def test_train_lock_sha_mismatch_and_missing_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = BuilderFixture(root, [_row("DatasetA", "DatasetA:train:1")])
            payload = json.loads(fixture.lock.read_text())
            payload["source"]["sha256"] = "0" * 64
            fixture.lock.write_text(json.dumps(payload))
            with self.assertRaisesRegex(DatasetBuildError, "SHA-256 mismatch"):
                verify_train_lock(fixture.project, fixture.lock, fixture.source_sha)
            payload["source"]["path"] = str(root / "missing.jsonl")
            fixture.lock.write_text(json.dumps(payload))
            with self.assertRaisesRegex(DatasetBuildError, "source is missing"):
                verify_train_lock(fixture.project, fixture.lock, fixture.source_sha)

    def test_authoritative_expected_sha_is_an_independent_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = BuilderFixture(root, [_row("DatasetA", "DatasetA:train:1")])
            with self.assertRaisesRegex(DatasetBuildError, "authoritative Train"):
                verify_train_lock(fixture.project, fixture.lock, "f" * 64)

    def test_formal_validation_or_test_path_is_rejected_without_opening(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forbidden = root / "Test"
            forbidden.mkdir()
            report = forbidden / "must-not-open.json"
            report.write_text("not-json")
            with self.assertRaisesRegex(DatasetBuildError, "Validation/Test"):
                verify_train_lock(root, report, None)

    def test_score_and_label_blind_source_mapping(self):
        forbidden = builder.FORBIDDEN_OUTPUT_KEYS
        candidates = [
            GuardedMapping(candidate, forbidden)
            for candidate in _balanced_candidates("blind")
        ]
        guarded = GuardedMapping(
            _row("DatasetA", "DatasetA:train:blind", candidates), forbidden
        )
        source_case = builder._source_case_from_row(guarded, 7)
        self.assertEqual("DatasetA:blind", source_case.canonical_underlying_case_id)
        self.assertNotIn("label", source_case.request)
        self.assertNotIn("selection_score", json.dumps(source_case.request))

    def test_empty_or_entirely_malformed_train_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = BuilderFixture(root / "empty", [])
            with self.assertRaisesRegex(DatasetBuildError, "contains no records"):
                empty.build(root / "empty-output")

            malformed = BuilderFixture(root / "malformed", [{"claim": "only"}])
            with self.assertRaisesRegex(DatasetBuildError, "required Frozen G1 request schema"):
                malformed.build(root / "malformed-output")

    def test_output_rejects_every_forbidden_selector_or_veracity_key(self):
        for key in sorted(builder.FORBIDDEN_OUTPUT_KEYS):
            with self.subTest(key=key):
                with self.assertRaises(DatasetBuildError):
                    assert_no_forbidden_output_fields({"nested": [{key: 1}]})

    def test_step25b_manifest_requires_exact_expected_five_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "selected_cases": [
                            {"canonical_underlying_case_id": item}
                            for item in sorted(EXPECTED_STEP25B_HELDOUT_IDS)
                        ]
                    }
                )
            )
            self.assertEqual(
                tuple(sorted(EXPECTED_STEP25B_HELDOUT_IDS)),
                load_step25b_heldout_ids(path),
            )
            payload = json.loads(path.read_text())
            payload["selected_cases"].pop()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(DatasetBuildError, "expected five"):
                load_step25b_heldout_ids(path)

    def test_all_step25b_and_cpac_identities_are_excluded_before_exposure(self):
        rows = [
            _row(identity.split(":")[0], identity)
            for identity in sorted(EXPECTED_STEP25B_HELDOUT_IDS)
        ]
        rows.append(_row("GroundLie360", "GroundLie360:train:13025004"))
        rows.append(_row("DatasetA", "DatasetA:train:eligible"))
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(Path(directory), rows)
            adapter = ControlledExposureAdapter()
            result = fixture.build(Path(directory) / "output", adapter)
            assignments = json.loads(
                (result.output_dir / "calibration_split_manifest.json").read_text()
            )["assignments"]
        self.assertEqual(1, len(adapter.calls))
        self.assertEqual("DatasetA:train:eligible", adapter.calls[0]["case_id"])
        self.assertFalse(
            set(result.build_report["heldout_case_ids"])
            & {
                item["canonical_underlying_case_id"]
                for item in assignments
            }
        )

    def test_canonical_identity_removes_only_exact_split_tokens(self):
        self.assertEqual(
            "TRUE-3MFact:10145403",
            canonicalize_underlying_case_id(
                "TRUE-3MFact", "TRUE-3MFact:train:10145403"
            ),
        )
        self.assertEqual(
            "GroundLie360:13025004",
            canonicalize_underlying_case_id(
                "GroundLie360", "GroundLie360:test:13025004"
            ),
        )
        self.assertEqual(
            "Data:phase2:101",
            canonicalize_underlying_case_id("Data:val", "phase2:101"),
        )
        self.assertNotEqual(
            canonicalize_underlying_case_id("DatasetA", "train:7"),
            canonicalize_underlying_case_id("DatasetB", "train:7"),
        )

    def test_duplicate_underlying_case_contributes_only_once(self):
        rows = [
            _row("DatasetA", "DatasetA:train:duplicate"),
            _row("DatasetA", "DatasetA:test:duplicate"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(Path(directory), rows)
            result = fixture.build(Path(directory) / "output")
        self.assertEqual(1, result.build_report["unique_underlying_case_count"])
        self.assertEqual(1, result.build_report["eligible_case_count"])

    def test_ambiguous_provenance_is_excluded_before_exposure_not_called_test_access(self):
        candidates = _balanced_candidates("ambiguous")
        candidates[0]["source"] = "artifact:/validation/segment"
        rows = [
            _row("DatasetA", "DatasetA:train:ambiguous", candidates),
            _row("DatasetA", "DatasetA:train:clean"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(Path(directory), rows)
            adapter = ControlledExposureAdapter()
            result = fixture.build(Path(directory) / "output", adapter)
        self.assertEqual(1, len(adapter.calls))
        self.assertEqual(1, result.build_report["ambiguous_provenance_exclusion_count"])
        self.assertFalse(result.build_report["formal_test_accessed"])

    def test_locked_groundlie_train_same_case_transcript_and_ocr_ids_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = BuilderFixture(root, [_groundlie_row("123")])
            adapter = ControlledExposureAdapter()
            result = _build_with_authoritative_train_lock(
                fixture, root / "output", adapter
            )
            card = (result.output_dir / "dataset_card.md").read_text(encoding="utf-8")
        self.assertEqual(1, len(adapter.calls))
        self.assertEqual(1, result.build_report["eligible_case_count"])
        self.assertEqual(
            0, result.build_report["ambiguous_provenance_exclusion_count"]
        )
        self.assertEqual(
            2,
            result.build_report[
                "groundlie_inherited_test_unit_id_accepted_count"
            ],
        )
        self.assertEqual(
            1,
            result.build_report["groundlie_inherited_test_unit_id_case_count"],
        )
        self.assertIn(
            "retain a historical `:test:` identifier token",
            card,
        )
        self.assertIn("exact same-case unit-ID pattern", card)
        self.assertFalse(result.build_report["formal_test_accessed"])

    def test_groundlie_inherited_unit_id_requires_same_underlying_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = BuilderFixture(
                root, [_groundlie_row("123", unit_case_id="999")]
            )
            adapter = ControlledExposureAdapter()
            result = _build_with_authoritative_train_lock(
                fixture, root / "output", adapter
            )
        self.assertEqual([], adapter.calls)
        self.assertEqual(
            1, result.build_report["ambiguous_provenance_exclusion_count"]
        )
        self.assertEqual(
            0,
            result.build_report[
                "groundlie_inherited_test_unit_id_accepted_count"
            ],
        )

    def test_groundlie_inherited_unit_id_requires_top_level_train_split(self):
        for split in ("test", "validation"):
            with self.subTest(split=split):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    fixture = BuilderFixture(
                        root, [_groundlie_row("123", split=split)]
                    )
                    adapter = ControlledExposureAdapter()
                    result = _build_with_authoritative_train_lock(
                        fixture, root / "output", adapter
                    )
                self.assertEqual([], adapter.calls)
                self.assertEqual(
                    1,
                    result.build_report["ambiguous_provenance_exclusion_count"],
                )
                self.assertEqual(
                    0,
                    result.build_report[
                        "groundlie_inherited_test_unit_id_accepted_count"
                    ],
                )

    def test_groundlie_inherited_unit_id_requires_exact_authoritative_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = BuilderFixture(root, [_groundlie_row("123")])
            self.assertNotEqual(
                builder.AUTHORITATIVE_TRAIN_SHA256, fixture.source_sha
            )
            adapter = ControlledExposureAdapter()
            result = fixture.build(root / "output", adapter)
        self.assertEqual([], adapter.calls)
        self.assertEqual(
            1, result.build_report["ambiguous_provenance_exclusion_count"]
        )
        self.assertEqual(
            0,
            result.build_report[
                "groundlie_inherited_test_unit_id_accepted_count"
            ],
        )

    def test_groundlie_exception_rejects_other_dataset_fields_and_markers(self):
        cases = (
            (
                "other_dataset",
                _row(
                    "TRUE-3MFact",
                    "TRUE-3MFact:train:123",
                    [
                        _candidate(
                            "TRUE-3MFact:test:123:transcript:0",
                            "transcript",
                            "Experts discuss public safety during this interview.",
                        ),
                        *_balanced_candidates("true-other")[1:],
                    ],
                    split="train",
                ),
            ),
            ("source_test", ("source", "artifact:test:segment")),
            ("snippet_path_test", ("snippet_path", "artifact/test/segment")),
            (
                "grounding_source_test",
                ("grounding", {"source": "artifact:test:segment"}),
            ),
            ("source_validation", ("source", "artifact:validation:segment")),
            (
                "snippet_path_validation",
                ("snippet_path", "artifact/validation/segment"),
            ),
        )
        for name, mutation in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    if name == "other_dataset":
                        row = mutation
                    else:
                        row = _groundlie_row("123")
                        field, value = mutation
                        row["candidate_units"][1][field] = value
                    fixture = BuilderFixture(root, [row])
                    adapter = ControlledExposureAdapter()
                    result = _build_with_authoritative_train_lock(
                        fixture, root / "output", adapter
                    )
                self.assertEqual([], adapter.calls)
                self.assertEqual(
                    1,
                    result.build_report["ambiguous_provenance_exclusion_count"],
                )
                self.assertEqual(
                    0,
                    result.build_report[
                        "groundlie_inherited_test_unit_id_accepted_count"
                    ],
                )

    def test_heldout_groundlie_precedes_inherited_unit_id_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = BuilderFixture(root, [_groundlie_row("13025004")])
            adapter = ControlledExposureAdapter()
            result = _build_with_authoritative_train_lock(
                fixture, root / "output", adapter
            )
        self.assertEqual([], adapter.calls)
        self.assertEqual(
            1, result.build_report["heldout_source_case_exclusion_count"]
        )
        self.assertEqual(
            0, result.build_report["ambiguous_provenance_exclusion_count"]
        )
        self.assertEqual(
            0,
            result.build_report[
                "groundlie_inherited_test_unit_id_accepted_count"
            ],
        )
        self.assertEqual(
            0,
            result.build_report["groundlie_inherited_test_unit_id_case_count"],
        )

    def test_all_required_ambiguous_provenance_markers_are_detected(self):
        for marker in (":test:", ":validation:", "/test/", "/validation/"):
            with self.subTest(marker=marker):
                candidate = _candidate(
                    "u", "ocr", "LONG ALPHABETIC OCR CONTENT HERE", source=f"a{marker}b"
                )
                self.assertTrue(builder.row_has_ambiguous_provenance([candidate]))

    def test_frozen_exposure_adapter_runs_before_anchor_selection_and_preserves_order(self):
        source = _balanced_candidates("exposure")

        def reverse_and_drop(_request, candidates):
            return [candidates[3], candidates[2], candidates[1], candidates[0]]

        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one", source)]
            )
            result = fixture.build(
                Path(directory) / "output", ControlledExposureAdapter(reverse_and_drop)
            )
            example = json.loads(
                (result.output_dir / "calibration_train.jsonl").read_text().splitlines()[0]
            )
        self.assertEqual(
            [source[3]["unit_id"], source[2]["unit_id"], source[1]["unit_id"], source[0]["unit_id"]],
            [item["unit_id"] for item in example["candidate_units"]],
        )
        self.assertEqual(4, example["model_exposed_candidate_count"])

    def test_truncated_or_rejected_candidates_cannot_become_anchors(self):
        source = _balanced_candidates("truncate")

        def remove_second_anchors(_request, candidates):
            return [candidates[0], candidates[2], candidates[4]]

        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one", source)]
            )
            result = fixture.build(
                Path(directory) / "output", ControlledExposureAdapter(remove_second_anchors)
            )
        self.assertEqual(0, result.build_report["eligible_case_count"])
        self.assertEqual(1, result.build_report["insufficient_transcript_anchor_count"])
        self.assertEqual(1, result.build_report["insufficient_ocr_anchor_count"])

    def test_missing_real_phase4a_implementation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one")]
            )
            with self.assertRaises(FrozenExposureUnavailableError):
                build_calibration_dataset(
                    project_root=fixture.project,
                    phase3a_train_lock_report=fixture.lock,
                    phase4a_config_path=fixture.config,
                    step25b_selected_manifest=fixture.manifest,
                    heldout_cases=("GroundLie360:13025004",),
                    output_dir=Path(directory) / "output",
                    exposure_adapter=None,
                    expected_train_sha256=fixture.source_sha,
                )

    def test_frozen_exposure_failure_for_every_case_fails_closed(self):
        class AlwaysFailingAdapter:
            def normalize(self, request):
                raise RuntimeError("fixture normalization failure")

        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one")]
            )
            with self.assertRaisesRegex(
                FrozenExposureUnavailableError, "every attempted Train case"
            ):
                fixture.build(Path(directory) / "output", AlwaysFailingAdapter())

    def test_cpac_strict_case_must_be_explicitly_held_out(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one")]
            )
            with self.assertRaisesRegex(DatasetBuildError, "CPAC strict-audit"):
                build_calibration_dataset(
                    project_root=fixture.project,
                    phase3a_train_lock_report=fixture.lock,
                    phase4a_config_path=fixture.config,
                    step25b_selected_manifest=fixture.manifest,
                    heldout_cases=("Dataset:other",),
                    output_dir=Path(directory) / "output",
                    exposure_adapter=ControlledExposureAdapter(),
                    expected_train_sha256=fixture.source_sha,
                )

    def test_real_exposure_loader_calls_injected_engine_normalizer_with_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            _write_phase4a_layout(project)
            config = root / "config.json"
            config.write_text('{"maximum_units_per_sample":4}\n', encoding="utf-8")
            adapter = builder.Phase4ANormalizationExposureAdapter.from_project_root(
                project, config
            )
            request = {
                "candidate_units": _balanced_candidates("actual-adapter"),
                "claim": "claim",
                "dataset": "DatasetA",
                "case_id": "case",
            }
            result = adapter.normalize(request)
        self.assertEqual(4, len(result.candidate_units))
        self.assertEqual(1, result.truncated_count)
        self.assertEqual(0, result.dropped_unsupported_count)

    def test_real_dicc_signature_receives_required_strict_visual_policy(self):
        calls = []

        def normalize_request(
            request,
            config,
            *,
            drop_unsupported_visual,
        ):
            calls.append(
                {
                    "request": request,
                    "config": config,
                    "drop_unsupported_visual": drop_unsupported_visual,
                }
            )
            result = dict(request)
            result["candidate_units"] = list(request["candidate_units"])
            result["truncated_unit_count"] = 0
            result["dropped_unsupported_count"] = 0
            return result

        config = {"maximum_units_per_sample": 24}
        adapter = builder.Phase4ANormalizationExposureAdapter(
            normalize_request, config
        )
        self.assertEqual([], calls)
        self.assertEqual(
            "config_keyword_with_strict_visual_policy", adapter._invocation
        )
        request = {
            "candidate_units": _balanced_candidates("strict-visual-policy"),
            "claim": "claim",
            "dataset": "DatasetA",
            "case_id": "case",
        }
        result = adapter.normalize(request)
        self.assertEqual(len(request["candidate_units"]), len(result.candidate_units))
        self.assertEqual(1, len(calls))
        self.assertIs(request, calls[0]["request"])
        self.assertIs(config, calls[0]["config"])
        self.assertIs(False, calls[0]["drop_unsupported_visual"])

    def test_real_exposure_loader_uses_exact_flat_import_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            _write_phase4a_layout(project)
            config = root / "config.json"
            config.write_text('{"maximum_units_per_sample":24}\n', encoding="utf-8")
            adapter = builder.Phase4ANormalizationExposureAdapter.from_project_root(
                project, config
            )
            import_path = adapter._normalize_request.__globals__[  # type: ignore[attr-defined]
                "IMPORT_PATH_SNAPSHOT"
            ]
        self.assertEqual(
            str(
                (
                    project / "MDU" / "scripts" / "clip12_phase3_common"
                ).resolve()
            ),
            import_path[0],
        )
        self.assertEqual(
            str(
                (
                    project
                    / "MDU"
                    / "scripts"
                    / "clip12_phase4a_inference_handoff"
                ).resolve()
            ),
            import_path[1],
        )

    def test_real_exposure_loader_does_not_rely_on_mdu_scripts_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            _write_phase4a_layout(project)
            config = root / "config.json"
            config.write_text('{"maximum_units_per_sample":24}\n', encoding="utf-8")
            adapter = builder.Phase4ANormalizationExposureAdapter.from_project_root(
                project, config
            )
            import_path = adapter._normalize_request.__globals__[  # type: ignore[attr-defined]
                "IMPORT_PATH_SNAPSHOT"
            ]
        self.assertNotIn(str((project / "MDU" / "scripts").resolve()), import_path)

    def test_real_exposure_loader_restores_sys_path_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            _write_phase4a_layout(project)
            config = root / "config.json"
            config.write_text('{"maximum_units_per_sample":24}\n', encoding="utf-8")
            original_sys_path = list(sys.path)
            builder.Phase4ANormalizationExposureAdapter.from_project_root(
                project, config
            )
            self.assertEqual(original_sys_path, sys.path)

    def test_real_exposure_loader_restores_sys_path_after_failed_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            _write_phase4a_layout(
                project,
                engine_source=(
                    "from clip12p3_common import common_marker\n"
                    "from clip12p3_model import model_marker\n"
                    "from clip12p4a_common import phase4a_marker\n"
                    "raise RuntimeError('fixture import failure')\n"
                ),
            )
            config = root / "config.json"
            config.write_text('{"maximum_units_per_sample":24}\n', encoding="utf-8")
            original_sys_path = list(sys.path)
            with self.assertRaisesRegex(
                FrozenExposureUnavailableError,
                "cannot import actual Phase4A normalization module",
            ):
                builder.Phase4ANormalizationExposureAdapter.from_project_root(
                    project, config
                )
            self.assertEqual(original_sys_path, sys.path)

    def test_real_exposure_loader_requires_every_real_dependency_file(self):
        required_names = (
            "clip12p3_common.py",
            "clip12p3_model.py",
            "clip12p4a_common.py",
            "clip12p4a_engine.py",
        )
        for missing_name in required_names:
            with self.subTest(missing_name=missing_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    project = root / "project"
                    files = _write_phase4a_layout(project)
                    files[missing_name].unlink()
                    config = root / "config.json"
                    config.write_text(
                        '{"maximum_units_per_sample":24}\n', encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        FrozenExposureUnavailableError, missing_name
                    ):
                        builder.Phase4ANormalizationExposureAdapter.from_project_root(
                            project, config
                        )

    def test_real_exposure_loader_missing_normalizer_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            _write_phase4a_layout(
                project,
                engine_source=(
                    "from clip12p3_common import common_marker\n"
                    "from clip12p3_model import model_marker\n"
                    "from clip12p4a_common import phase4a_marker\n"
                ),
            )
            config = root / "config.json"
            config.write_text('{"maximum_units_per_sample":24}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                FrozenExposureUnavailableError,
                "Phase4A normalize_request is unavailable",
            ):
                builder.Phase4ANormalizationExposureAdapter.from_project_root(
                    project, config
                )

    def test_real_exposure_loader_accepts_request_only_pure_normalizer(self):
        adapter = builder.Phase4ANormalizationExposureAdapter(
            lambda request: request, {}
        )
        request = {
            "candidate_units": _balanced_candidates("request-only"),
            "claim": "claim",
            "dataset": "DatasetA",
            "case_id": "case",
        }
        result = adapter.normalize(request)
        self.assertEqual(len(request["candidate_units"]), len(result.candidate_units))

    def test_real_exposure_loader_rejects_incompatible_signature(self):
        with self.assertRaises(FrozenExposureUnavailableError):
            builder.Phase4ANormalizationExposureAdapter(lambda: {}, {})

    def test_duplicate_exposed_candidate_ids_fail_closed(self):
        source = _balanced_candidates("duplicate-id")

        def duplicate(_request, candidates):
            return candidates + [copy.deepcopy(candidates[0])]

        class DuplicateAdapter:
            def normalize(self, request):
                candidates = duplicate(request, list(request["candidate_units"]))
                return ExposureResult(tuple(candidates), len(candidates), 0, 0)

        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one", source)]
            )
            with self.assertRaisesRegex(DatasetBuildError, "duplicate model-exposed"):
                fixture.build(Path(directory) / "output", DuplicateAdapter())

    def test_text_normalization_is_nfkc_quote_stable_and_deterministic(self):
        value = "  ＡＢＣ  “quoted”  ‘words’  "
        self.assertEqual('ABC "quoted" \'words\'', normalize_text(value))
        self.assertEqual(lexical_signature(value), lexical_signature(normalize_text(value)))

    def test_transcript_quality_exact_boundaries_and_rejections(self):
        valid = "Four clear alphabetic tokens appear in this transcript"
        self.assertIsNone(transcript_quality_reason(valid))
        self.assertEqual(
            "transcript_character_length_below_20",
            transcript_quality_reason("one two three four"),
        )
        self.assertEqual(
            "transcript_character_length_above_240",
            transcript_quality_reason("word " * 60),
        )
        self.assertEqual("transcript_url_only", transcript_quality_reason("https://example.com/path/to/resource"))
        self.assertEqual(
            "transcript_handle_only",
            transcript_quality_reason("@firsthandle @secondhandle @thirdhandle @fourthhandle"),
        )

    def test_ocr_quality_exact_boundaries_and_numeric_noise(self):
        valid = "AMERICAN CONSERVATIVE UNION CPAC EVENT 2018"
        self.assertIsNone(ocr_quality_reason(valid))
        self.assertEqual(
            "ocr_character_length_below_20",
            ocr_quality_reason("WORDS ARE SHORT"),
        )
        self.assertEqual(
            "ocr_character_length_above_240", ocr_quality_reason("ALPHA word " * 30)
        )
        self.assertEqual(
            "ocr_numeric_timestamp_noise_above_0_50",
            ocr_quality_reason("LONGALPHABET word 12 13 14"),
        )
        self.assertEqual(
            "ocr_url_only", ocr_quality_reason("https://example.com/long/ocr/resource")
        )

    def test_only_transcript_and_ocr_can_be_anchor_modalities(self):
        self.assertEqual("TRANSCRIPT", candidate_anchor_modality("transcript", "text"))
        self.assertEqual("OCR", candidate_anchor_modality("ocr", "text"))
        self.assertEqual("OCR", candidate_anchor_modality("text", "ocr"))
        for unit_type in ("evidence", "rationale", "metadata", "title", "text"):
            self.assertIsNone(candidate_anchor_modality(unit_type, "text"))

    def test_exact_two_plus_two_examples_per_eligible_case(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one")]
            )
            result = fixture.build(Path(directory) / "output")
            lines = (result.output_dir / "calibration_train.jsonl").read_text().splitlines()
            examples = [json.loads(line) for line in lines]
        self.assertEqual(4, len(examples))
        self.assertEqual(2, sum(item["expected_modality"] == "OCR" for item in examples))
        self.assertEqual(2, sum(item["expected_modality"] == "TRANSCRIPT" for item in examples))

    def test_duplicate_anchor_signatures_do_not_consume_two_slots(self):
        candidates = _balanced_candidates("dedup")
        candidates[1]["text"] = candidates[0]["text"].upper()
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one", candidates)]
            )
            result = fixture.build(Path(directory) / "output")
        self.assertEqual(0, result.build_report["eligible_case_count"])
        self.assertEqual(1, result.build_report["insufficient_transcript_anchor_count"])

    def test_anchor_group_order_is_deterministic_under_input_reversal(self):
        rows_a = [_row("DatasetA", "DatasetA:train:one")]
        rows_b = copy.deepcopy(rows_a)
        rows_b[0]["candidate_units"].reverse()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = BuilderFixture(root / "a", rows_a).build(root / "out-a")
            second = BuilderFixture(root / "b", rows_b).build(root / "out-b")
            first_manifest = json.loads(
                (first.output_dir / "eligible_case_manifest.json").read_text()
            )
            second_manifest = json.loads(
                (second.output_dir / "eligible_case_manifest.json").read_text()
            )
        first_case = first_manifest["eligible_cases"][0]
        second_case = second_manifest["eligible_cases"][0]
        self.assertEqual(
            first_case["selected_ocr_anchor_unit_ids"], second_case["selected_ocr_anchor_unit_ids"]
        )
        self.assertEqual(
            first_case["selected_transcript_anchor_unit_ids"],
            second_case["selected_transcript_anchor_unit_ids"],
        )

    def test_positive_targets_are_strict_same_modality_direct_grounding(self):
        candidates = _balanced_candidates("positive")
        anchor_text = candidates[2]["text"]
        candidates[3]["text"] = anchor_text + " LIVE"
        candidates.insert(3, _candidate("positive-o-duplicate", "ocr", anchor_text.lower()))
        candidates.insert(4, _candidate("positive-o-contained", "ocr", anchor_text + " LIVE"))
        candidates.insert(5, _candidate("positive-t-same", "transcript", anchor_text))
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one", candidates)]
            )
            result = fixture.build(Path(directory) / "output")
            examples = [
                json.loads(line)
                for line in (result.output_dir / "calibration_train.jsonl").read_text().splitlines()
            ]
        example = next(
            item
            for item in examples
            if item["expected_modality"] == "OCR"
            and item["anchor_text"].casefold() == anchor_text.casefold()
        )
        self.assertIn(candidates[2]["unit_id"], example["positive_unit_ids"])
        self.assertIn("positive-o-duplicate", example["positive_unit_ids"])
        self.assertIn("positive-o-contained", example["positive_unit_ids"])
        self.assertNotIn("positive-t-same", example["positive_unit_ids"])
        targets = {item["unit_id"]: item["relevance_target"] for item in example["candidate_units"]}
        self.assertEqual(0, targets[candidates[-1]["unit_id"]])

    def test_case_disjoint_split_is_deterministic_per_dataset_and_about_eighty_twenty(self):
        rows = []
        for dataset in ("DatasetA", "DatasetB"):
            for index in range(10):
                rows.append(_row(dataset, f"{dataset}:train:{index}"))
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(Path(directory), rows)
            first = fixture.build(Path(directory) / "first")
            second = fixture.build(Path(directory) / "second")
            first_manifest = json.loads(
                (first.output_dir / "calibration_split_manifest.json").read_text()
            )
            second_manifest = json.loads(
                (second.output_dir / "calibration_split_manifest.json").read_text()
            )
        self.assertEqual(first_manifest, second_manifest)
        for dataset in ("DatasetA", "DatasetB"):
            splits = [
                item["calibration_split"]
                for item in first_manifest["assignments"]
                if item["source_dataset"] == dataset
            ]
            self.assertEqual(8, splits.count("train"))
            self.assertEqual(2, splits.count("dev"))
        train = {
            item["canonical_underlying_case_id"]
            for item in first_manifest["assignments"]
            if item["calibration_split"] == "train"
        }
        dev = {
            item["canonical_underlying_case_id"]
            for item in first_manifest["assignments"]
            if item["calibration_split"] == "dev"
        }
        self.assertFalse(train & dev)

    def test_output_schema_has_provenance_targets_unique_ids_and_train_dev_names(self):
        rows = [_row("DatasetA", f"DatasetA:train:{index}") for index in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(Path(directory), rows)
            result = fixture.build(Path(directory) / "output")
            examples = []
            for name in ("calibration_train.jsonl", "calibration_dev.jsonl"):
                examples.extend(
                    json.loads(line)
                    for line in (result.output_dir / name).read_text().splitlines()
                )
        self.assertEqual(20, len(examples))
        for example in examples:
            self.assertIn(example["calibration_split"], {"train", "dev"})
            self.assertIn(example["expected_modality"], {"OCR", "TRANSCRIPT"})
            self.assertIn("train_variant_sha256", example["source_provenance"])
            ids = [item["unit_id"] for item in example["candidate_units"]]
            self.assertEqual(len(ids), len(set(ids)))
            positives = [
                item["unit_id"]
                for item in example["candidate_units"]
                if item["relevance_target"] == 1
            ]
            self.assertEqual(positives, example["positive_unit_ids"])
            assert_no_forbidden_output_fields(example)

    def test_identical_input_produces_byte_identical_authoritative_outputs_and_sidecars(self):
        rows = [_row("DatasetA", f"DatasetA:train:{index}") for index in range(6)]
        names = (
            "source_inventory.json",
            "heldout_exclusions.json",
            "eligible_case_manifest.json",
            "eligible_case_manifest.sha256",
            "calibration_split_manifest.json",
            "calibration_split_manifest.sha256",
            "calibration_train.jsonl",
            "calibration_train.sha256",
            "calibration_dev.jsonl",
            "calibration_dev.sha256",
            "build_report.json",
            "dataset_card.md",
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(Path(directory), rows)
            first = fixture.build(Path(directory) / "first")
            second = fixture.build(Path(directory) / "second")
            for name in names:
                self.assertEqual(
                    (first.output_dir / name).read_bytes(),
                    (second.output_dir / name).read_bytes(),
                    name,
                )
            for artifact, digest in first.build_report["artifact_sha256"].items():
                self.assertEqual(
                    digest, hashlib.sha256((first.output_dir / artifact).read_bytes()).hexdigest()
                )
            for stem in (
                "eligible_case_manifest",
                "calibration_split_manifest",
                "calibration_train",
                "calibration_dev",
            ):
                digest = (first.output_dir / f"{stem}.sha256").read_text().strip()
                suffix = ".jsonl" if stem.startswith("calibration_t") or stem == "calibration_dev" else ".json"
                self.assertEqual(
                    digest,
                    hashlib.sha256((first.output_dir / f"{stem}{suffix}").read_bytes()).hexdigest(),
                )

    def test_eligible_and_split_manifests_are_frozen_before_target_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = BuilderFixture(
                root, [_row("DatasetA", "DatasetA:train:freeze-order")]
            )
            output = root / "output"
            original = builder.make_calibration_example

            def guarded(*args, **kwargs):
                self.assertTrue((output / "eligible_case_manifest.json").is_file())
                self.assertTrue((output / "eligible_case_manifest.sha256").is_file())
                self.assertTrue((output / "calibration_split_manifest.json").is_file())
                self.assertTrue((output / "calibration_split_manifest.sha256").is_file())
                return original(*args, **kwargs)

            with patch.object(builder, "make_calibration_example", side_effect=guarded):
                fixture.build(output)

    def test_build_report_declares_all_closed_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = BuilderFixture(
                Path(directory), [_row("DatasetA", "DatasetA:train:one")]
            )
            report = fixture.build(Path(directory) / "output").build_report
        self.assertFalse(report["selection_outputs_inspected"])
        self.assertFalse(report["veracity_labels_inspected"])
        self.assertFalse(report["formal_validation_accessed"])
        self.assertFalse(report["formal_test_accessed"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["checkpoint_loaded"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["production_or_model_code_changed"])


if __name__ == "__main__":
    unittest.main()
