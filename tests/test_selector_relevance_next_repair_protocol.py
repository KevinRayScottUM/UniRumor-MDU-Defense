from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest import mock

from scripts.selector_relevance_next_repair_protocol.exclusion_contract import (
    REQUIRED_COUNTS, ExclusionLock, ExclusionManifest, identity_sha256,
)
from scripts.selector_relevance_next_repair_protocol.protocol import (
    PREREGISTRATION_FILENAME, PREREGISTRATION_PATH, load_preregistration,
    protocol_preflight, validate_coverage, validate_development_cohort,
    validate_preregistration,
)
from scripts.selector_relevance_next_repair_protocol.run_protocol_preflight import build_parser, main
from scripts.selector_relevance_next_repair_protocol.schemas import (
    Candidate, DevelopmentCase, FrozenTrainCase, ProtocolError,
)


def manifest(name, identities):
    identities = tuple(identities)
    return ExclusionManifest(
        name, identities, identity_sha256(identities),
        hashlib.sha256(("synthetic-source:" + name).encode()).hexdigest(),
    )


def exclusion_lock(overrides=None, future=()):
    overrides = overrides or {}
    manifests = []
    for name, count in REQUIRED_COUNTS.items():
        identities = [f"GroundLie360:synthetic-{name}-{index}" for index in range(count)]
        if name in overrides:
            identities[0] = overrides[name]
        manifests.append(manifest(name, identities))
    manifests.extend(future)
    return ExclusionLock(tuple(manifests))


def synthetic_case(dataset, index, *, count=6):
    pairs = (("evidence", "text"), ("title_span", "text"),
             ("transcript", "text"), ("ocr", "ocr"))
    return FrozenTrainCase(
        f"{dataset}:synthetic-case-{index:03d}", dataset,
        f"  Synthetic natural claim {index}: e\u0301 / 中文\n ",
        tuple(Candidate(f"unit-{position}", *pairs[position % 4],
                        f" Synthetic original candidate {position}\t ")
              for position in range(count)),
    )


def development_fixture(lock):
    source = tuple(synthetic_case(dataset, index) for dataset in
                   ("GroundLie360", "TRUE-3MFact") for index in range(80))
    proposed = []
    # Independent reference implementation of the preregistered identity-only ordering.
    def order(case, purpose):
        message = ["step2.6r-3c1-natural-pairwise-v1", purpose,
                   case.dataset, case.canonical_case_id]
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest(), case.canonical_case_id
    for dataset in ("GroundLie360", "TRUE-3MFact"):
        available = [case for case in source if case.dataset == dataset
                     and case.canonical_case_id not in lock.all_identities]
        chosen = sorted(available, key=lambda case: order(case, "repair-development"))[:60]
        chosen = sorted(chosen, key=lambda case: order(case, "repair-split"))
        proposed.extend(DevelopmentCase(case, "repair_train" if index < 48 else "repair_dev")
                        for index, case in enumerate(chosen))
    return source, tuple(proposed)


def final_labels(proposed, train_evaluable, dev_evaluable):
    remaining = {"repair_train": train_evaluable, "repair_dev": dev_evaluable}
    result = {}
    for row in proposed:
        positive = remaining[row.repair_split] > 0
        remaining[row.repair_split] -= int(positive)
        result[row.case.canonical_case_id] = {
            unit.unit_id: "DIRECT" if index == 0 and positive else "RELATED"
            for index, unit in enumerate(row.case.candidate_units)
        }
    return result


class ExclusionContractTests(unittest.TestCase):
    def test_complete_30_identity_quarantine_cannot_be_omitted_or_truncated(self):
        lock = exclusion_lock()
        with self.assertRaises(ProtocolError):
            ExclusionLock(tuple(item for item in lock.manifests if item.name != "revealed_audit"))
        with self.assertRaises(ProtocolError):
            manifest("revealed_audit", tuple(f"TRUE-3MFact:synthetic-{i}" for i in range(29)))

    def test_identity_digest_detects_mutation(self):
        entry = exclusion_lock().manifests[1]
        with self.assertRaisesRegex(ProtocolError, "SHA-256 mismatch"):
            replace(entry, canonical_case_ids=("TRUE-3MFact:changed", *entry.canonical_case_ids[1:]))

    def test_duplicate_ids_and_missing_source_hash_fail(self):
        with self.assertRaises(ProtocolError):
            manifest("revealed_audit", ("TRUE-3MFact:same",) * 30)
        with self.assertRaises(ProtocolError):
            replace(exclusion_lock().manifests[0], source_artifact_sha256="")

    def test_membership_and_effective_counts_report_overlaps_without_double_counting(self):
        shared = "GroundLie360:shared"
        lock = exclusion_lock({name: shared for name in REQUIRED_COUNTS},
                              future=(manifest("future:extra", (shared, "TRUE-3MFact:other")),))
        report = lock.accounting((shared, "TRUE-3MFact:other", "TRUE-3MFact:allowed"))
        rows = report["categories"]
        self.assertEqual([row["membership_count"] for row in rows], [1, 1, 1, 1, 2])
        self.assertEqual([row["effective_sequential_count"] for row in rows], [1, 0, 0, 0, 1])
        self.assertEqual(report["unique_excluded_union_count"], 2)
        self.assertEqual(report["remaining_count"], 1)
        self.assertEqual(sum(row["effective_sequential_count"] for row in rows), 2)
        self.assertTrue(all(row["overlap_count"] == 1 for row in report["pairwise_overlap_counts"]))

    def test_future_manifests_are_required_and_ordered(self):
        lock = exclusion_lock(future=(manifest("future:z", ("TRUE-3MFact:z",)),
                                     manifest("future:a", ("TRUE-3MFact:a",))))
        self.assertEqual([item.name for item in lock.ordered][-2:], ["future:a", "future:z"])
        with self.assertRaises(ProtocolError):
            replace(lock, expected_future_manifest_names=("future:missing",))
        with self.assertRaises(ProtocolError):
            lock.reject_overlap(("TRUE-3MFact:a",))

    def test_no_override_parameter(self):
        with self.assertRaises(TypeError):
            ExclusionLock(exclusion_lock().manifests, allow_overlap=True)


class DevelopmentContractTests(unittest.TestCase):
    def setUp(self):
        self.lock = exclusion_lock()
        self.source, self.proposed = development_fixture(self.lock)

    def check(self, proposed=None, source=None, lock=None):
        return validate_development_cohort(
            self.source if source is None else source,
            self.proposed if proposed is None else proposed,
            self.lock if lock is None else lock,
        )

    def test_exact_120_deterministic_balanced_manifest_passes(self):
        result = self.check()
        self.assertEqual(result["case_count"], 120)
        self.assertEqual(result["dataset_counts"], {"GroundLie360": 60, "TRUE-3MFact": 60})
        self.assertEqual(result["split_counts"], {"repair_train": 96, "repair_dev": 24})
        self.assertTrue(result["original_claims_and_candidate_fields_unchanged"])

    def test_source_inventory_order_does_not_affect_selection(self):
        self.assertEqual(self.check(), self.check(source=tuple(reversed(self.source))))

    def test_old_30_audit_overlap_rejected(self):
        self._overlap("revealed_audit")

    def test_old_calibration_overlap_rejected(self):
        self._overlap("prior_calibration")

    def test_sealed_six_overlap_rejected(self):
        self._overlap("sealed_challenge")

    def test_stage_a_overlap_rejected(self):
        self._overlap("stage_a_replay")

    def _overlap(self, name):
        lock = exclusion_lock({name: self.proposed[0].case.canonical_case_id})
        with self.assertRaisesRegex(ProtocolError, "exclusion/quarantine"):
            self.check(lock=lock)

    def test_future_exclusion_overlap_rejected(self):
        entry = manifest("future:new", (self.proposed[0].case.canonical_case_id,))
        with self.assertRaisesRegex(ProtocolError, "exclusion/quarantine"):
            self.check(lock=exclusion_lock(future=(entry,)))

    def test_exactly_120_cases_required(self):
        with self.assertRaisesRegex(ProtocolError, "exactly 120"):
            self.check(proposed=self.proposed[:-1])

    def test_60_60_dataset_balance_required(self):
        selected = {row.case.canonical_case_id for row in self.proposed}
        extra = next(case for case in self.source if case.dataset == "GroundLie360"
                     and case.canonical_case_id not in selected)
        changed = list(self.proposed)
        changed[60] = DevelopmentCase(extra, changed[60].repair_split)
        with self.assertRaisesRegex(ProtocolError, "60/60"):
            self.check(proposed=changed)

    def test_exact_96_24_split_required(self):
        changed = list(self.proposed)
        changed[0] = replace(changed[0], repair_split="repair_dev")
        with self.assertRaisesRegex(ProtocolError, "96/24"):
            self.check(proposed=changed)

    def test_48_48_train_and_12_12_dev_balance_required(self):
        changed = list(self.proposed)
        changed[0] = replace(changed[0], repair_split="repair_dev")
        changed[108] = replace(changed[108], repair_split="repair_train")
        with self.assertRaisesRegex(ProtocolError, "48/48 Train and 12/12 Dev"):
            self.check(proposed=changed)

    def test_train_dev_overlap_rejected(self):
        changed = list(self.proposed)
        changed[48] = DevelopmentCase(changed[0].case, "repair_dev")
        with self.assertRaisesRegex(ProtocolError, "Train/Dev overlap"):
            self.check(proposed=changed)

    def test_balanced_but_nonpreregistered_split_rejected(self):
        changed = list(self.proposed)
        changed[0] = replace(changed[0], repair_split="repair_dev")
        changed[48] = replace(changed[48], repair_split="repair_train")
        with self.assertRaisesRegex(ProtocolError, "deterministic hash"):
            self.check(proposed=changed)

    def test_balanced_but_nonpreregistered_membership_rejected(self):
        selected = {row.case.canonical_case_id for row in self.proposed}
        extra = next(case for case in self.source if case.dataset == "GroundLie360"
                     and case.canonical_case_id not in selected)
        changed = list(self.proposed)
        changed[0] = replace(changed[0], case=extra)
        with self.assertRaisesRegex(ProtocolError, "deterministic hash"):
            self.check(proposed=changed)

    def test_original_claim_and_candidate_fields_order_are_preserved(self):
        row = self.proposed[0]
        for changed_case in (
            replace(row.case, original_claim=row.case.original_claim.strip()),
            replace(row.case, original_claim='The relevant content states "neutral".'),
            replace(row.case, candidate_units=tuple(reversed(row.case.candidate_units))),
            replace(row.case, candidate_units=(replace(row.case.candidate_units[0],
                text="changed"), *row.case.candidate_units[1:])),
            replace(row.case, candidate_units=(replace(row.case.candidate_units[0],
                unit_type="transcript"), *row.case.candidate_units[1:])),
        ):
            with self.subTest(changed_case=changed_case):
                with self.assertRaisesRegex(ProtocolError, "differs from source"):
                    self.check(proposed=(replace(row, case=changed_case), *self.proposed[1:]))
        self.assertEqual(row.case.original_claim.encode(), self.source[
            self.source.index(row.case)].original_claim.encode())
        with self.assertRaises(FrozenInstanceError):
            row.case.original_claim = "changed"

    def test_candidate_count_6_to_24(self):
        for count in (6, 24):
            self.assertEqual(len(synthetic_case("GroundLie360", 0, count=count).candidate_units), count)
        for count in (0, 5, 25):
            with self.assertRaises(ProtocolError):
                synthetic_case("GroundLie360", 0, count=count)

    def test_all_frozen_pairs_accepted_visual_rejected(self):
        for pair in (("evidence", "text"), ("title_span", "text"),
                     ("transcript", "text"), ("ocr", "ocr")):
            self.assertEqual(Candidate("id", *pair, "original").text, "original")
        for pair in (("visual", "text"), ("image", "text"),
                     ("evidence", "ocr"), ("ocr", "text")):
            with self.assertRaises(ProtocolError):
                Candidate("id", *pair, "original")

    def test_formal_validation_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "Formal Validation/Test"):
            replace(self.source[0], source_partition="Validation")

    def test_formal_test_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "Formal Validation/Test"):
            replace(self.source[0], source_partition="Test")

    def test_insufficient_unexcluded_inventory_has_no_fallback(self):
        allowed_inventory = tuple(row.case for row in self.proposed)
        entry = manifest("future:block", (self.proposed[0].case.canonical_case_id,))
        with self.assertRaises(ProtocolError):
            self.check(source=allowed_inventory, lock=exclusion_lock(future=(entry,)))


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.lock = exclusion_lock()
        self.source, self.proposed = development_fixture(self.lock)

    def coverage(self, train, dev):
        return validate_coverage(self.source, self.proposed, self.lock,
                                 final_labels(self.proposed, train, dev))

    def test_overall_96_passes_its_condition_but_all_conditions_are_required(self):
        report = self.coverage(76, 20)
        self.assertEqual(report["evaluable_counts"]["overall"], 96)
        self.assertTrue(report["checks"]["overall"])
        self.assertFalse(report["coverage_pass"])
        self.assertTrue(report["repair_training_blocked"])

    def test_overall_95_fails(self):
        self.assertFalse(self.coverage(76, 19)["checks"]["overall"])

    def test_train_77_passes_and_76_fails(self):
        self.assertTrue(self.coverage(77, 20)["checks"]["repair_train"])
        self.assertFalse(self.coverage(76, 20)["checks"]["repair_train"])

    def test_dev_20_passes_and_19_fails(self):
        self.assertTrue(self.coverage(77, 20)["checks"]["repair_dev"])
        self.assertFalse(self.coverage(77, 19)["checks"]["repair_dev"])

    def test_zero_direct_retained_and_no_resampling(self):
        report = self.coverage(77, 20)
        self.assertTrue(report["coverage_pass"])
        self.assertEqual(report["retained_case_count"], 120)
        self.assertEqual(len(report["zero_direct_case_ids"]), 23)
        self.assertFalse(report["resampling_performed"])
        self.assertFalse(report["zero_direct_in_ranking_denominator"])

    def test_dropping_zero_direct_cases_or_units_rejected(self):
        labels = final_labels(self.proposed, 77, 20)
        zero = next(identity for identity, values in labels.items() if "DIRECT" not in values.values())
        changed = copy.deepcopy(labels)
        changed.pop(zero)
        with self.assertRaisesRegex(ProtocolError, "every frozen development case"):
            validate_coverage(self.source, self.proposed, self.lock, changed)
        changed = copy.deepcopy(labels)
        changed[zero].pop("unit-0")
        with self.assertRaisesRegex(ProtocolError, "every frozen candidate"):
            validate_coverage(self.source, self.proposed, self.lock, changed)


class FrozenPreregistrationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_preregistration()

    def test_static_preregistration_and_prior_valid_fail(self):
        validate_preregistration(self.protocol)
        self.assertEqual(self.protocol["implementation_revision"], "step2.6r-3c1-v1")
        prior = self.protocol["prior_result"]
        self.assertEqual(prior["interpretation"], "VALID SCIENTIFIC FAIL")
        self.assertTrue(prior["scientific_result_valid"])
        self.assertFalse(prior["repair_verification_pass"])
        self.assertTrue(prior["deployment_remains_blocked"])

    def test_entire_preregistration_is_exact_and_rejects_mutation(self):
        for section in self.protocol:
            changed = copy.deepcopy(self.protocol)
            if isinstance(changed[section], dict):
                changed[section]["override"] = True
            else:
                changed[section] = "changed"
            with self.subTest(section=section):
                with self.assertRaises(ProtocolError):
                    validate_preregistration(changed)
        changed = copy.deepcopy(self.protocol)
        changed["training"]["maximum_epochs"] = 10.0
        with self.assertRaises(ProtocolError):
            validate_preregistration(changed)

    def test_exact_pairwise_loss_and_forbidden_weighting(self):
        objective = self.protocol["objective"]
        self.assertEqual(objective["P"], "all DIRECT units in the case")
        self.assertEqual(objective["N"], "all non-DIRECT units in the case")
        self.assertEqual(objective["d"], "s_positive - s_negative")
        self.assertEqual(objective["pair_loss"], "softplus(-d)")
        self.assertEqual(objective["case_loss"], "mean over all positive-negative pairs in the case")
        self.assertEqual(objective["batch_loss"], "mean over evaluable cases in the batch")
        self.assertEqual(objective["zero_direct_loss_contribution"], "none")
        self.assertEqual(objective["forbidden_supervision_and_weighting"], [
            "veracity_margin", "veracity_labels", "modality_weighting", "dataset_weighting",
            "ocr_bonus", "transcript_penalty", "source_specific_offset", "confidence_weighting",
        ])

    def test_only_selection_head_may_train(self):
        boundary = self.protocol["model_boundary"]
        self.assertEqual(boundary["trainable_only"], ["selection_head.weight", "selection_head.bias"])
        self.assertEqual(boundary["frozen"], ["encoder", "veracity_head"])
        self.assertEqual(boundary["base_checkpoint_sha256"],
                         "b694f2d4bb5ba6f72dd8a001bd984d46853546f2a85858a812f2496af1f1a0b9")
        self.assertTrue(boundary["prediction_path_unchanged"])
        self.assertTrue(boundary["top_k_explanation_only"])

    def test_natural_claim_encoding_and_annotation(self):
        encoding = self.protocol["encoding"]
        self.assertEqual(encoding["sequence_a"], "original natural claim")
        self.assertEqual(encoding["sequence_b"], "[UNIT_TYPE=<unit_type>] [MODALITY=<modality>] <candidate text>")
        self.assertFalse(encoding["neutral_synthetic_anchor_templates_permitted"])
        annotation = self.protocol["annotation"]
        self.assertEqual(annotation["binary_mapping"],
                         {"DIRECT": 1, "RELATED": 0, "IRRELEVANT": 0, "UNREADABLE": 0})
        self.assertEqual(annotation["all_four_class_disagreements"], "Independent Reviewer C adjudication")
        self.assertFalse(annotation["confidence_tie_breaking_permitted"])

    def test_seeds_hyperparameters_and_dev_selection_order_exact(self):
        training = self.protocol["training"]
        self.assertEqual(training["seeds"], [42, 43, 44])
        self.assertEqual(training["maximum_epochs"], 10)
        self.assertEqual(training["optimizer"], "AdamW")
        self.assertEqual(training["learning_rate"], 1e-3)
        self.assertEqual(training["weight_decay"], 0)
        self.assertEqual(training["early_stopping_patience"], 2)
        metrics = ["Dev MRR descending", "Dev NDCG@5 descending", "Dev Recall@5 descending"]
        self.assertEqual(training["best_epoch_order"], metrics + ["earlier epoch"])
        self.assertEqual(training["deployment_candidate_seed_order"], metrics + ["lower seed number"])
        self.assertEqual(training["model_selection_source"], "new 24-case repair_dev split only")

    def test_old_audit_cannot_be_any_selection_source(self):
        quarantine = self.protocol["quarantine"]
        self.assertTrue(quarantine["permanent"])
        self.assertFalse(quarantine["override_permitted"])
        self.assertEqual(set(quarantine["forbidden_uses"]), {
            "training", "calibration", "hyperparameter_tuning", "seed_selection",
            "epoch_selection", "loss_selection", "architecture_selection",
            "threshold_tuning", "model_selection", "future_acceptance_gating",
        })
        changed = copy.deepcopy(self.protocol)
        changed["training"]["model_selection_source"] = "revealed old 30-case audit"
        with self.assertRaises(ProtocolError):
            validate_preregistration(changed)

    def test_new_invariance_and_future_audit_stage_order(self):
        replay = self.protocol["new_invariance"]
        self.assertEqual(replay["prediction_mismatch_count_required"], 0)
        self.assertFalse(replay["relevance_labels_used"])
        self.assertEqual(replay["verify_unchanged"], ["candidate_ids", "candidate_order",
                         "encoder_outputs", "veracity_logits", "sample_probabilities", "sample_prediction"])
        audit = self.protocol["future_audit"]
        self.assertEqual(audit["construction_requires"], ["repair development complete",
                         "best seed frozen", "model artifact frozen", "prediction invariance PASS"])
        self.assertEqual(audit["dataset_counts"], {"GroundLie360": 15, "TRUE-3MFact": 15})
        self.assertFalse(audit["model_scores_viewed_before_final_gold_freeze"])
        self.assertTrue(audit["one_shot_scoring"])

    def test_future_gate_matches_3b3_frozen_numeric_conditions(self):
        from scripts.selector_relevance_independent_eval.source_loader import _frozen_protocol
        old = _frozen_protocol()["repair_verification_gate"]
        gate = self.protocol["future_one_shot_gate"]
        self.assertEqual(gate["minimum_evaluable_case_count"], old["minimum_evaluable_case_count"])
        self.assertEqual(gate["minimum_max_of_delta_mrr_and_delta_ndcg_at_5"],
                         old["minimum_absolute_mrr_or_ndcg_at_5_improvement"])
        self.assertEqual(gate["minimum_delta_recall_at_5"], -old["maximum_recall_at_5_decrease"])
        self.assertEqual(gate["minimum_groundlie_delta_mrr"], -old["maximum_groundlie_mrr_decrease"])
        self.assertEqual(gate["minimum_true3m_delta_mrr"], -old["maximum_true3m_mrr_decrease"])
        self.assertTrue(gate["overall_mrr_strictly_greater_than_original"])
        self.assertTrue(gate["overall_ndcg_at_5_strictly_greater_than_original"])
        self.assertTrue(gate["prediction_architecture_unchanged"])
        self.assertTrue(gate["all_conditions_required"])
        for name in ("cpac_gate", "regression_count_gate", "modality_gate"):
            self.assertFalse(gate[name])

    def test_no_automatic_deployment(self):
        self.assertFalse(self.protocol["deployment"]["automatic_after_future_pass"])
        self.assertTrue(self.protocol["deployment"]["separate_explicit_downstream_authorization_required"])


class ProtocolPreflightBoundaryTests(unittest.TestCase):
    def test_preflight_reports_protocol_only_without_model_data_or_scores(self):
        report = protocol_preflight()
        self.assertEqual(report["status"], "NEXT_REPAIR_PROTOCOL_PREFLIGHT_PASS")
        self.assertEqual(report["real_exclusion_lock_status"], "REQUIRED_NOT_SUPPLIED")
        for field in ("real_cohort_ready", "real_dataset_built", "reviewer_packets_created",
                      "model_loaded", "training_performed", "selector_scoring_performed"):
            self.assertFalse(report[field])

    def test_fresh_process_preflight_imports_no_torch_or_model_runtime(self):
        script = '''
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'paddle', 'numpy'}:
        raise AssertionError('model dependency import forbidden: ' + name)
    if name.startswith('scripts.selector_relevance_gate'):
        raise AssertionError('historical runtime import forbidden')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from scripts.selector_relevance_next_repair_protocol.run_protocol_preflight import main
assert main([]) == 0
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["model_loaded"])

    def test_preflight_reads_only_static_preregistration_and_writes_nothing(self):
        opened = []
        original = Path.read_text
        def reader(path, *args, **kwargs):
            opened.append(path.resolve())
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "read_text", reader), mock.patch.object(
            Path, "write_text", side_effect=AssertionError("no writes")
        ), mock.patch.object(Path, "write_bytes", side_effect=AssertionError("no writes")):
            protocol_preflight()
        self.assertEqual(opened, [PREREGISTRATION_PATH.resolve()])

    def test_formal_paths_rejected_before_open(self):
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("must not open")):
            for part in ("Validation", "Test", "FormalValidation", "FormalTest"):
                with self.subTest(part=part), self.assertRaises(ProtocolError):
                    load_preregistration(Path("outputs") / part / PREREGISTRATION_FILENAME)

    def test_ranking_output_filename_rejected_before_open(self):
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("must not open")):
            with self.assertRaises(ProtocolError):
                load_preregistration(Path("outputs") / "ranking_scores.jsonl")

    def test_duplicate_json_keys_and_modified_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / PREREGISTRATION_FILENAME
            path.write_text('{"scope":"PROTOCOL_ONLY","scope":"changed"}', encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "duplicate"):
                load_preregistration(path)
            path.write_text('{}', encoding="utf-8")
            with self.assertRaises(ProtocolError):
                load_preregistration(path)

    def test_cli_exposes_no_execution_or_override_options(self):
        options = {name for action in build_parser()._actions for name in action.option_strings}
        self.assertEqual(options, {"-h", "--help", "--preregistration"})
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([]), 0)
        self.assertEqual(json.loads(output.getvalue())["scope"], "PROTOCOL_ONLY")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--preregistration", "ranking_scores.jsonl"]), 2)


if __name__ == "__main__":
    unittest.main()
