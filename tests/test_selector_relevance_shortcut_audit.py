from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from scripts.selector_relevance_shortcut_audit import audit
from scripts.selector_relevance_shortcut_audit.audit import (
    AuditInputError,
    ExpectedCounts,
    build_template_analysis,
    classify_shortcut_risk,
    inspect_encoding_contract,
    load_calibration_artifacts,
    parse_original_template,
    predict_claim_only_modality,
    recommend_training_action,
    run_shortcut_audit,
)
from scripts.selector_relevance_shortcut_audit.run_audit import build_parser


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


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _example(
    example_id: str,
    modality: str,
    split: str,
    case_id: str,
    *,
    claim: Optional[str] = None,
) -> Dict[str, Any]:
    anchor_text = (
        "PUBLIC SAFETY NOTICE DISPLAYED ON SCREEN"
        if modality == audit.OCR
        else "Experts discuss public safety during this interview"
    )
    prefix = audit.OCR_PREFIX if modality == audit.OCR else audit.TRANSCRIPT_PREFIX
    anchor_id = f"{example_id}-positive"
    return {
        "schema_version": 1,
        "calibration_example_id": example_id,
        "source_dataset": "FixtureDataset",
        "source_case_id": case_id,
        "canonical_underlying_case_id": case_id,
        "calibration_split": split,
        "expected_modality": modality,
        "claim": claim if claim is not None else f'{prefix}{anchor_text}".',
        "anchor_unit_id": anchor_id,
        "anchor_text": anchor_text,
        "positive_unit_ids": [anchor_id],
        "model_exposed_candidate_count": 2,
        "candidate_units": [
            {
                "unit_id": anchor_id,
                "unit_type": "ocr" if modality == audit.OCR else "transcript",
                "modality": "ocr" if modality == audit.OCR else "text",
                "text": anchor_text,
                "relevance_target": 1,
            },
            {
                "unit_id": f"{example_id}-negative",
                "unit_type": "text",
                "modality": "text",
                "text": "An unrelated candidate unit remains a negative.",
                "relevance_target": 0,
            },
        ],
        "source_provenance": {
            "train_variant_sha256": "1" * 64,
            "phase4a_config_sha256": "2" * 64,
            "source_row_index": 0,
        },
    }


def _write_calibration_fixture(root: Path) -> Path:
    calibration = root / "calibration"
    calibration.mkdir()
    train = [_example("example-ocr", audit.OCR, "train", "FixtureDataset:1")]
    dev = [
        _example(
            "example-transcript",
            audit.TRANSCRIPT,
            "dev",
            "FixtureDataset:2",
        )
    ]
    split_manifest = {
        "schema_version": 1,
        "split_unit": "canonical_underlying_case_id",
        "assignments": [
            {
                "canonical_underlying_case_id": "FixtureDataset:1",
                "source_dataset": "FixtureDataset",
                "calibration_split": "train",
            },
            {
                "canonical_underlying_case_id": "FixtureDataset:2",
                "source_dataset": "FixtureDataset",
                "calibration_split": "dev",
            },
        ],
    }
    eligible_manifest = {
        "schema_version": 1,
        "eligible_case_count": 2,
        "eligible_cases": [
            {"canonical_underlying_case_id": "FixtureDataset:1"},
            {"canonical_underlying_case_id": "FixtureDataset:2"},
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
        (calibration / name).write_bytes(content)
        hashes[name] = _sha256(content)
    for stem in ("calibration_train", "calibration_dev"):
        (calibration / f"{stem}.sha256").write_text(
            hashes[f"{stem}.jsonl"] + "\n", encoding="utf-8"
        )
    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "implementation_revision": audit.CALIBRATION_REVISION,
        "calibration_train_example_count": 1,
        "calibration_dev_example_count": 1,
        "ocr_example_count": 1,
        "transcript_example_count": 1,
        "artifact_sha256": hashes,
        "selection_outputs_inspected": False,
        "veracity_labels_inspected": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "training_started": False,
    }
    (calibration / "build_report.json").write_bytes(_json_bytes(report))
    return calibration


def _refresh_report_hash(calibration: Path, name: str) -> None:
    report_path = calibration / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact_sha256"][name] = audit.sha256_file(calibration / name)
    report_path.write_bytes(_json_bytes(report))


def _write_encoding_fixture(root: Path, *, explicit: bool = False) -> Path:
    phase3_common = root / "MDU/scripts/clip12_phase3_common"
    phase3_final = root / "MDU/scripts/clip12_phase3a_final_fit"
    phase4a = root / "MDU/scripts/clip12_phase4a_inference_handoff"
    configs = root / "MDU/configs"
    for directory in (phase3_common, phase3_final, phase4a, configs):
        directory.mkdir(parents=True, exist_ok=True)
    if explicit:
        pair_source = (
            "class MDUSelectorVerifier:\n"
            "    pass\n\n"
            "def encode_pair(tokenizer, claim, unit_text, unit_type, max_length):\n"
            "    serialized_unit = f'{unit_type}: {unit_text}'\n"
            "    return tokenizer(claim, serialized_unit, padding='max_length', "
            "truncation=True, max_length=max_length)\n"
        )
    else:
        pair_source = (
            "class MDUSelectorVerifier:\n"
            "    pass\n\n"
            "def encode_pair(tokenizer, claim, unit_text, max_length):\n"
            "    return tokenizer(claim, unit_text, padding='max_length', "
            "truncation=True, max_length=max_length)\n"
        )
    (phase3_common / "clip12p3_model.py").write_text(pair_source, encoding="utf-8")
    (phase3_final / "clip12p3a_final_fit.py").write_text(
        "from clip12p3_model import MDUSelectorVerifier\n\n"
        "def build_model():\n"
        "    return MDUSelectorVerifier()\n",
        encoding="utf-8",
    )
    (phase4a / "clip12p4a_engine.py").write_text(
        "def inference_handoff(candidate_units):\n"
        "    return list(candidate_units)\n",
        encoding="utf-8",
    )
    (configs / "clip12_phase4a_frozen_g1_inference_handoff.json").write_bytes(
        _json_bytes({"model": {"max_length": 256}, "max_units": 24})
    )
    return root


def _write_resolved_encoding_fixture(root: Path) -> Path:
    project = _write_encoding_fixture(root)
    phase3_common = project / "MDU/scripts/clip12_phase3_common/clip12p3_model.py"
    phase3_common.write_text(
        "class MDUSelectorVerifier:\n"
        "    pass\n\n"
        "class Batch:\n"
        "    def __init__(self, encoded, unit_ids):\n"
        "        self.encoded = encoded\n"
        "        self.unit_ids = unit_ids\n\n"
        "class MDUDataset:\n"
        "    def __getitem__(self, index):\n"
        "        row = self.rows[index]\n"
        "        return {\n"
        "            'claim': row['claim'],\n"
        "            'dataset': row['dataset'],\n"
        "            'units': ordered_units(row['candidate_units']),\n"
        "        }\n\n"
        "def collator(tokenizer, max_length):\n"
        "    def collate(items):\n"
        "        pair_texts = []\n"
        "        unit_ids = []\n"
        "        datasets = []\n"
        "        for item in items:\n"
        "            datasets.append(item['dataset'])\n"
        "            for unit in item['units']:\n"
        "                prefix = (\n"
        "                    f\"[UNIT_TYPE={unit.get('unit_type')}] \"\n"
        "                    f\"[MODALITY={unit.get('modality')}]\"\n"
        "                )\n"
        "                pair_texts.append((\n"
        "                    item['claim'],\n"
        "                    prefix + ' ' + normalize_text(unit.get('text')),\n"
        "                ))\n"
        "                unit_ids.append(unit.get('unit_id'))\n"
        "        encoded = tokenizer(\n"
        "            [pair[0] for pair in pair_texts],\n"
        "            [pair[1] for pair in pair_texts],\n"
        "            padding=True,\n"
        "            truncation=True,\n"
        "            max_length=max_length,\n"
        "            return_tensors='pt',\n"
        "        )\n"
        "        return Batch(encoded, unit_ids)\n"
        "    return collate\n",
        encoding="utf-8",
    )
    phase3_final = (
        project / "MDU/scripts/clip12_phase3a_final_fit/clip12p3a_final_fit.py"
    )
    phase3_final.write_text(
        "from clip12p3_model import MDUSelectorVerifier, collator\n\n"
        "def final_fit(tokenizer, max_length):\n"
        "    model = MDUSelectorVerifier()\n"
        "    return model, collator(tokenizer, max_length)\n",
        encoding="utf-8",
    )
    phase4a = (
        project
        / "MDU/scripts/clip12_phase4a_inference_handoff/clip12p4a_engine.py"
    )
    phase4a.write_text(
        "class Phase4AEngine:\n"
        "    def encode(self, normalized, units, max_length):\n"
        "        source_dataset = normalized.get('dataset')\n"
        "        batch_unit_ids = [unit.get('unit_id') for unit in units]\n"
        "        encoded = self.tokenizer(\n"
        "            [normalized['claim']] * len(units),\n"
        "            [\n"
        "                f\"[UNIT_TYPE={unit.get('unit_type')}] \"\n"
        "                f\"[MODALITY={unit.get('modality')}] \"\n"
        "                f\"{normalize_text(unit.get('text'))}\"\n"
        "                for unit in units\n"
        "            ],\n"
        "            padding=True,\n"
        "            truncation=True,\n"
        "            max_length=max_length,\n"
        "            return_tensors='pt',\n"
        "        )\n"
        "        return encoded, source_dataset, batch_unit_ids\n",
        encoding="utf-8",
    )
    return project


class TemplateAndControlTests(unittest.TestCase):
    def test_original_templates_and_claim_only_predictor(self) -> None:
        ocr = parse_original_template('The on-screen text reads "NOTICE".')
        transcript = parse_original_template('The speaker says "hello".')
        self.assertIsNotNone(ocr)
        self.assertIsNotNone(transcript)
        self.assertEqual(audit.OCR, ocr.modality)
        self.assertEqual("NOTICE", ocr.anchor_text)
        self.assertEqual(audit.TRANSCRIPT, transcript.modality)
        self.assertEqual("hello", transcript.anchor_text)
        self.assertEqual(
            audit.OCR,
            predict_claim_only_modality('The on-screen text reads "NOTICE".'),
        )
        self.assertEqual(
            audit.TRANSCRIPT,
            predict_claim_only_modality('The speaker says "hello".'),
        )

    def test_malformed_and_unknown_templates_do_not_classify(self) -> None:
        self.assertIsNone(parse_original_template('The speaker says "unterminated'))
        self.assertEqual(
            audit.UNKNOWN,
            predict_claim_only_modality('The relevant content states "NOTICE".'),
        )
        self.assertEqual(audit.UNKNOWN, predict_claim_only_modality(None))

    def test_accuracy_is_computed_from_examples(self) -> None:
        examples = [
            _example("a", audit.OCR, "train", "FixtureDataset:a"),
            _example("b", audit.TRANSCRIPT, "dev", "FixtureDataset:b"),
            _example(
                "c",
                audit.OCR,
                "train",
                "FixtureDataset:c",
                claim='A neutral claim states "PUBLIC SAFETY NOTICE DISPLAYED ON SCREEN".',
            ),
        ]
        report, _ = build_template_analysis(examples)
        self.assertAlmostEqual(2 / 3, report["claim_only_template_modality_accuracy"])
        self.assertEqual(2, report["claim_only_correct_count"])
        self.assertAlmostEqual(2 / 3, report["original_template_conformity_rate"])

    def test_controls_preserve_lexical_content_and_all_registered_targets(self) -> None:
        examples = [
            _example("b", audit.TRANSCRIPT, "dev", "FixtureDataset:b"),
            _example("a", audit.OCR, "train", "FixtureDataset:a"),
        ]
        report, manifest = build_template_analysis(examples)
        self.assertEqual('The relevant content states "<TEXT>".', report["neutral_control_template"])
        self.assertEqual(2, report["neutral_control_constructable_count"])
        self.assertEqual(2, report["swapped_control_constructable_count"])
        self.assertEqual(0, report["control_target_invariance_failures"])
        rows = {item["calibration_example_id"]: item for item in manifest}
        self.assertEqual(
            'The relevant content states "PUBLIC SAFETY NOTICE DISPLAYED ON SCREEN".',
            rows["a"]["neutral_claim"],
        )
        self.assertEqual(
            'The speaker says "PUBLIC SAFETY NOTICE DISPLAYED ON SCREEN".',
            rows["a"]["swapped_template_claim"],
        )
        self.assertEqual(
            'The on-screen text reads "Experts discuss public safety during this interview".',
            rows["b"]["swapped_template_claim"],
        )
        for row in rows.values():
            self.assertTrue(row["anchor_text_unchanged"])
            self.assertTrue(row["canonical_underlying_case_id_unchanged"])
            self.assertTrue(row["calibration_split_unchanged"])
            self.assertTrue(row["anchor_unit_id_unchanged"])
            self.assertTrue(row["positive_unit_ids_unchanged"])
            self.assertTrue(row["candidate_unit_ids_unchanged"])
            self.assertTrue(row["relevance_targets_unchanged"])
            self.assertTrue(row["targets_unchanged"])


class ClassificationTests(unittest.TestCase):
    def test_pre_registered_shortcut_classifications(self) -> None:
        self.assertEqual(
            "HIGH_TEMPLATE_MODALITY_SHORTCUT_RISK",
            classify_shortcut_risk(1.0, "EXPLICIT"),
        )
        self.assertEqual(
            "MODERATE_TEMPLATE_MODALITY_SHORTCUT_RISK",
            classify_shortcut_risk(1.0, "IMPLICIT_ONLY"),
        )
        self.assertEqual(
            "LOW_TEMPLATE_MODALITY_SHORTCUT_RISK",
            classify_shortcut_risk(0.75, "IMPLICIT_ONLY"),
        )
        self.assertEqual("INCONCLUSIVE", classify_shortcut_risk(1.0, audit.UNKNOWN))
        self.assertEqual(
            "INCONCLUSIVE",
            classify_shortcut_risk(1.0, "EXPLICIT", encoding_contract_verified=False),
        )

    def test_pre_registered_training_recommendations(self) -> None:
        expected = {
            "HIGH_TEMPLATE_MODALITY_SHORTCUT_RISK": "REQUIRE_TEMPLATE_NEUTRAL_CALIBRATION_BEFORE_TRAINING",
            "MODERATE_TEMPLATE_MODALITY_SHORTCUT_RISK": "REQUIRE_TEMPLATE_NEUTRAL_CALIBRATION_BEFORE_TRAINING",
            "LOW_TEMPLATE_MODALITY_SHORTCUT_RISK": "ORIGINAL_TEMPLATE_TRAINING_ACCEPTABLE",
            "INCONCLUSIVE": "REQUIRE_FURTHER_ENCODING_AUDIT",
        }
        for risk, recommendation in expected.items():
            with self.subTest(risk=risk):
                self.assertEqual(recommendation, recommend_training_action(risk))


class ArtifactIntegrityTests(unittest.TestCase):
    def test_sha_sidecar_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calibration = _write_calibration_fixture(Path(temporary))
            (calibration / "calibration_train.sha256").write_text(
                "0" * 64 + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(AuditInputError, "SHA sidecar mismatch"):
                load_calibration_artifacts(calibration)

    def test_malformed_jsonl_fails_closed_after_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calibration = _write_calibration_fixture(Path(temporary))
            malformed = b'{"calibration_example_id":\n'
            (calibration / "calibration_train.jsonl").write_bytes(malformed)
            digest = _sha256(malformed)
            (calibration / "calibration_train.sha256").write_text(
                digest + "\n", encoding="utf-8"
            )
            _refresh_report_hash(calibration, "calibration_train.jsonl")
            with self.assertRaisesRegex(AuditInputError, "row 0 is malformed"):
                load_calibration_artifacts(calibration)

    def test_non_object_eligible_case_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calibration = _write_calibration_fixture(Path(temporary))
            eligible_path = calibration / "eligible_case_manifest.json"
            payload = json.loads(eligible_path.read_text(encoding="utf-8"))
            payload["eligible_cases"].append("not-an-object")
            eligible_path.write_bytes(_json_bytes(payload))
            _refresh_report_hash(calibration, "eligible_case_manifest.json")
            with self.assertRaisesRegex(AuditInputError, "must be an object"):
                load_calibration_artifacts(calibration)

    def test_formal_validation_and_test_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for restricted_name in ("Formal_Validation", "Formal Test"):
                restricted = root / restricted_name / "calibration"
                with self.subTest(restricted_name=restricted_name):
                    with self.assertRaisesRegex(
                        AuditInputError, "must not reference Formal Validation/Test"
                    ):
                        load_calibration_artifacts(restricted)


class EncodingContractTests(unittest.TestCase):
    def test_static_implicit_only_sequence_pair_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_encoding_fixture(Path(temporary))
            contract = inspect_encoding_contract(project)
        self.assertTrue(contract["encoding_contract_verified"])
        self.assertFalse(contract["runtime_trace_available"])
        self.assertEqual("MDUSelectorVerifier", contract["model_class_used"])
        self.assertEqual("SEQUENCE_PAIR", contract["claim_unit_encoding_style"])
        self.assertTrue(contract["claim_text_serialized"])
        self.assertTrue(contract["unit_text_serialized"])
        self.assertFalse(contract["unit_type_serialized"])
        self.assertFalse(contract["modality_serialized"])
        self.assertEqual("IMPLICIT_ONLY", contract["unit_modality_encoding"])
        self.assertEqual(256, contract["maximum_sequence_length"])
        self.assertEqual("max_length", contract["maximum_sequence_length_expression"])
        self.assertEqual("longest_first", contract["truncation_strategy"])
        self.assertEqual("both sequences as needed", contract["truncatable_side"])
        self.assertEqual("max_length", contract["padding"])
        self.assertEqual(4, len(contract["source_file_sha256"]))

    def test_static_explicit_unit_type_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_encoding_fixture(Path(temporary), explicit=True)
            contract = inspect_encoding_contract(project)
        self.assertTrue(contract["encoding_contract_verified"])
        self.assertTrue(contract["unit_type_serialized"])
        self.assertEqual("EXPLICIT", contract["unit_modality_encoding"])
        self.assertIn("serialized_unit", contract["unit_serialization_expression"])

    def test_keyword_sequence_pair_is_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_encoding_fixture(Path(temporary))
            source = project / "MDU/scripts/clip12_phase3_common/clip12p3_model.py"
            source.write_text(
                "class MDUSelectorVerifier:\n"
                "    pass\n\n"
                "def encode_pair(tokenizer, claim, unit_text, max_length):\n"
                "    return tokenizer(text=claim, text_pair=unit_text, "
                "padding='max_length', truncation='only_second', "
                "max_length=max_length)\n",
                encoding="utf-8",
            )
            contract = inspect_encoding_contract(project)
        self.assertTrue(contract["encoding_contract_verified"])
        self.assertEqual("SEQUENCE_PAIR", contract["claim_unit_encoding_style"])
        self.assertEqual("only_second", contract["truncation_strategy"])
        self.assertEqual("unit sequence only", contract["truncatable_side"])

    def test_final_phase3a_encoding_call_has_static_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_encoding_fixture(Path(temporary))
            source = (
                project
                / "MDU/scripts/clip12_phase3a_final_fit/clip12p3a_final_fit.py"
            )
            source.write_text(
                "from clip12p3_model import MDUSelectorVerifier\n\n"
                "def encode_final(tokenizer, claim, unit_text, modality, max_length):\n"
                "    final_unit = f'{modality}: {unit_text}'\n"
                "    return tokenizer(claim, final_unit, truncation=True, "
                "max_length=max_length)\n\n"
                "def build_model():\n"
                "    return MDUSelectorVerifier()\n",
                encoding="utf-8",
            )
            contract = inspect_encoding_contract(project)
        evidence = contract["encoding_contract_evidence"][0]
        self.assertEqual(
            "MDU/scripts/clip12_phase3a_final_fit/clip12p3a_final_fit.py",
            evidence["file"],
        )
        self.assertEqual(2, contract["encoding_contract_candidate_count"])
        self.assertEqual("EXPLICIT", contract["unit_modality_encoding"])

    def test_real_phase3a_collator_indirect_pair_flow_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_resolved_encoding_fixture(Path(temporary))
            contract = inspect_encoding_contract(project)
        self.assertTrue(contract["encoding_contract_verified"])
        self.assertEqual("SEQUENCE_PAIR", contract["claim_unit_encoding_style"])
        self.assertTrue(contract["claim_text_serialized"])
        self.assertTrue(contract["unit_text_serialized"])
        self.assertTrue(contract["unit_type_serialized"])
        self.assertTrue(contract["modality_serialized"])
        self.assertFalse(contract["dataset_serialized"])
        self.assertFalse(contract["unit_id_serialized"])
        self.assertEqual("EXPLICIT", contract["unit_modality_encoding"])
        self.assertIs(True, contract["padding"])
        self.assertIs(True, contract["truncation"])
        self.assertIs(True, contract["truncation_expression"])
        self.assertEqual(256, contract["maximum_sequence_length"])
        self.assertEqual("max_length", contract["maximum_sequence_length_expression"])
        evidence = contract["encoding_contract_evidence"][0]
        self.assertEqual(
            "MDU/scripts/clip12_phase3_common/clip12p3_model.py",
            evidence["file"],
        )
        expression = contract["unit_serialization_expression"]
        self.assertIn("append_tuple[1]", expression)
        self.assertIn("UNIT_TYPE", expression)
        self.assertIn("MODALITY", expression)
        self.assertIn("unit.get('text')", expression)
        risk = classify_shortcut_risk(1.0, contract["unit_modality_encoding"])
        self.assertEqual("HIGH_TEMPLATE_MODALITY_SHORTCUT_RISK", risk)
        self.assertEqual(
            "REQUIRE_TEMPLATE_NEUTRAL_CALIBRATION_BEFORE_TRAINING",
            recommend_training_action(risk),
        )

    def test_real_phase4a_comprehension_pair_is_independently_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_resolved_encoding_fixture(Path(temporary))
            common = project / "MDU/scripts/clip12_phase3_common/clip12p3_model.py"
            common.write_text(
                "class MDUSelectorVerifier:\n"
                "    pass\n",
                encoding="utf-8",
            )
            contract = inspect_encoding_contract(project)
        self.assertTrue(contract["encoding_contract_verified"])
        self.assertEqual("SEQUENCE_PAIR", contract["claim_unit_encoding_style"])
        self.assertTrue(contract["claim_text_serialized"])
        self.assertTrue(contract["unit_text_serialized"])
        self.assertTrue(contract["unit_type_serialized"])
        self.assertTrue(contract["modality_serialized"])
        self.assertFalse(contract["dataset_serialized"])
        self.assertFalse(contract["unit_id_serialized"])
        self.assertEqual("EXPLICIT", contract["unit_modality_encoding"])
        self.assertIs(True, contract["padding"])
        self.assertIs(True, contract["truncation"])
        self.assertIs(True, contract["truncation_expression"])
        self.assertEqual(256, contract["maximum_sequence_length"])
        self.assertEqual(
            "MDU/scripts/clip12_phase4a_inference_handoff/clip12p4a_engine.py",
            contract["encoding_contract_evidence"][0]["file"],
        )

    def test_genuinely_unknown_encoding_remains_unknown_and_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_encoding_fixture(Path(temporary))
            common = project / "MDU/scripts/clip12_phase3_common/clip12p3_model.py"
            common.write_text(
                "class MDUSelectorVerifier:\n"
                "    pass\n\n"
                "def opaque_batch(tokenizer, opaque_features):\n"
                "    return tokenizer(opaque_features, padding=True)\n",
                encoding="utf-8",
            )
            contract = inspect_encoding_contract(project)
        self.assertFalse(contract["encoding_contract_verified"])
        self.assertEqual(audit.UNKNOWN, contract["claim_unit_encoding_style"])
        self.assertEqual(audit.UNKNOWN, contract["unit_modality_encoding"])
        self.assertEqual(
            "INCONCLUSIVE",
            classify_shortcut_risk(
                1.0,
                contract["unit_modality_encoding"],
                encoding_contract_verified=contract["encoding_contract_verified"],
            ),
        )

    def test_mixed_pair_append_roles_do_not_create_a_broad_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_resolved_encoding_fixture(Path(temporary))
            common = project / "MDU/scripts/clip12_phase3_common/clip12p3_model.py"
            source = common.read_text(encoding="utf-8")
            source = source.replace(
                "                unit_ids.append(unit.get('unit_id'))\n",
                "                pair_texts.append((item['dataset'], unit.get('unit_id')))\n"
                "                unit_ids.append(unit.get('unit_id'))\n",
            )
            common.write_text(source, encoding="utf-8")
            phase4a = (
                project
                / "MDU/scripts/clip12_phase4a_inference_handoff/clip12p4a_engine.py"
            )
            phase4a.write_text(
                "def opaque_inference(tokenizer, opaque_features):\n"
                "    return tokenizer(opaque_features)\n",
                encoding="utf-8",
            )
            contract = inspect_encoding_contract(project)
        self.assertFalse(contract["encoding_contract_verified"])
        self.assertEqual(audit.UNKNOWN, contract["unit_modality_encoding"])

    def test_missing_authoritative_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_encoding_fixture(Path(temporary))
            (
                project
                / "MDU/scripts/clip12_phase3a_final_fit/clip12p3a_final_fit.py"
            ).unlink()
            with self.assertRaisesRegex(AuditInputError, "source is missing"):
                inspect_encoding_contract(project)

    def test_source_is_parsed_not_imported_or_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_encoding_fixture(Path(temporary))
            source = project / "MDU/scripts/clip12_phase3_common/clip12p3_model.py"
            source.write_text(
                "raise RuntimeError('must never execute')\n" + source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            contract = inspect_encoding_contract(project)
        self.assertTrue(contract["encoding_contract_verified"])


class EndToEndAndBoundaryTests(unittest.TestCase):
    def test_cli_requires_registered_paths(self) -> None:
        args = build_parser().parse_args(
            [
                "--project-root",
                "/project",
                "--calibration-dir",
                "/calibration",
                "--output-dir",
                "/output",
            ]
        )
        self.assertEqual(Path("/project"), args.project_root)
        self.assertEqual(Path("/calibration"), args.calibration_dir)
        self.assertEqual(Path("/output"), args.output_dir)

    def test_complete_audit_writes_only_diagnostic_outputs_and_preserves_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = _write_calibration_fixture(root)
            project = _write_encoding_fixture(root / "project")
            output = root / "audit-output"
            before = {
                name: (calibration / name).read_bytes()
                for name in audit._REQUIRED_CALIBRATION_FILES
            }
            report = run_shortcut_audit(
                project_root=project,
                calibration_dir=calibration,
                output_dir=output,
                expected_counts=ExpectedCounts(total=2, ocr=1, transcript=1),
            )
            after = {
                name: (calibration / name).read_bytes()
                for name in audit._REQUIRED_CALIBRATION_FILES
            }
            self.assertEqual(before, after)
            self.assertEqual("COMPLETED", report["status"])
            self.assertEqual(
                "MODERATE_TEMPLATE_MODALITY_SHORTCUT_RISK",
                report["shortcut_risk_classification"],
            )
            self.assertEqual(
                "REQUIRE_TEMPLATE_NEUTRAL_CALIBRATION_BEFORE_TRAINING",
                report["training_authorization_recommendation"],
            )
            expected_outputs = {
                "encoding_contract.json",
                "template_leakage_report.json",
                "template_control_manifest.jsonl",
                "template_control_manifest.sha256",
                "shortcut_audit_report.json",
                "README_AUDIT.md",
            }
            self.assertEqual(expected_outputs, {path.name for path in output.iterdir()})
            manifest = output / "template_control_manifest.jsonl"
            self.assertEqual(
                audit.sha256_file(manifest),
                (output / "template_control_manifest.sha256").read_text(
                    encoding="utf-8"
                ).strip(),
            )
            for field in (
                "selection_outputs_inspected",
                "veracity_labels_inspected",
                "formal_validation_accessed",
                "formal_test_accessed",
                "model_loaded",
                "checkpoint_loaded",
                "training_started",
                "production_or_model_code_changed",
            ):
                self.assertFalse(report[field])

    def test_nonempty_output_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = _write_calibration_fixture(root)
            project = _write_encoding_fixture(root / "project")
            output = root / "audit-output"
            output.mkdir()
            (output / "existing.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(AuditInputError, "absent or empty"):
                run_shortcut_audit(
                    project_root=project,
                    calibration_dir=calibration,
                    output_dir=output,
                    expected_counts=ExpectedCounts(total=2, ocr=1, transcript=1),
                )
            self.assertEqual(
                "preserve", (output / "existing.txt").read_text(encoding="utf-8")
            )

    def test_audit_source_has_no_model_runtime_imports(self) -> None:
        source = Path(audit.__file__).read_text(encoding="utf-8")
        forbidden_imports = (
            "FrozenG1Runner",
            "MDUSelectorVerifier",
            "transformers",
            "torch",
            "Phase4A",
        )
        import_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        for forbidden in forbidden_imports:
            self.assertFalse(
                any(forbidden in line for line in import_lines),
                msg=f"forbidden runtime import found: {forbidden}",
            )

    def test_template_audit_never_reads_forbidden_scientific_outputs(self) -> None:
        forbidden = {
            "selection_score",
            "selector_score",
            "logits",
            "prediction",
            "label",
            "veracity_label",
        }

        class GuardedDict(dict):
            def __getitem__(self, key: str) -> Any:
                if key in forbidden:
                    raise AssertionError(f"forbidden field accessed: {key}")
                return super().__getitem__(key)

            def get(self, key: str, default: Any = None) -> Any:
                if key in forbidden:
                    raise AssertionError(f"forbidden field accessed: {key}")
                return super().get(key, default)

        example = _example("guarded", audit.OCR, "train", "FixtureDataset:guarded")
        example.update({key: object() for key in forbidden})
        example["candidate_units"] = [
            GuardedDict({**candidate, **{key: object() for key in forbidden}})
            for candidate in example["candidate_units"]
        ]
        report, manifest = build_template_analysis([GuardedDict(example)])
        self.assertEqual(1, report["total_example_count"])
        self.assertEqual(1, len(manifest))


if __name__ == "__main__":
    unittest.main()
