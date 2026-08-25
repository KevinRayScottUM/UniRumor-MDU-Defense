import json
import tempfile
import unittest
from pathlib import Path

from schemas import RuntimeUnit, SourceType
from scripts.selector_fidelity_audit.audit import (
    AuditInputError,
    ProbeDefinition,
    classify_audit,
    compute_probe_metrics,
    derive_ranked_units,
    load_candidate_pool,
    load_probe_manifest,
)


def scored_unit(
    unit_id: str,
    source_type: SourceType,
    score: float,
) -> RuntimeUnit:
    return RuntimeUnit(
        unit_id=unit_id,
        source_type=source_type,
        text=f"Evidence for {unit_id}",
        eligible_for_frozen_g1=True,
        selection_score=score,
        logits={"fake": score - 0.1, "real": score + 0.1},
    )


class SelectorFidelityAuditTests(unittest.TestCase):
    def test_ranking_preserves_exposure_order_and_uses_stable_tie_break(self) -> None:
        units = [
            scored_unit("ocr-low", SourceType.OCR, -0.2),
            scored_unit("transcript-first", SourceType.TRANSCRIPT, 0.7),
            scored_unit("transcript-tied", SourceType.TRANSCRIPT, 0.7),
            scored_unit("ocr-middle", SourceType.OCR, 0.1),
        ]

        ranked = derive_ranked_units(units, ["transcript-tied", "ocr-middle"])

        self.assertEqual(
            ["ocr-low", "transcript-first", "transcript-tied", "ocr-middle"],
            [item.unit_id for item in ranked],
        )
        self.assertEqual([4, 1, 2, 3], [item.selection_rank for item in ranked])
        self.assertEqual(
            [False, False, True, True],
            [item.top_k_member for item in ranked],
        )

    def test_metrics_flag_direct_grounding_miss_without_changing_scores(self) -> None:
        units = [
            scored_unit(f"transcript-{index}", SourceType.TRANSCRIPT, 1.0 - index / 10)
            for index in range(5)
        ] + [scored_unit("ocr-direct", SourceType.OCR, -0.9)]
        ranked = derive_ranked_units(
            units,
            [f"transcript-{index}" for index in range(5)],
        )
        probe = ProbeDefinition(
            probe_id="ocr-probe",
            claim="The text appears on screen.",
            expected_modality="OCR",
            expected_relevant_unit_ids=("ocr-direct",),
            direct_grounding_unit_ids=("ocr-direct",),
            annotation_status="audited",
            annotation_basis="Synthetic audit-only fixture.",
        )

        metrics = compute_probe_metrics(probe, ranked)

        self.assertEqual(0.0, metrics["recall_at_5"])
        self.assertAlmostEqual(1.0 / 6.0, metrics["mrr"])
        self.assertEqual(0.0, metrics["ndcg_at_5"])
        self.assertEqual(6, metrics["highest_relevant_unit_rank"])
        self.assertEqual(6, metrics["best_expected_modality_rank"])
        self.assertFalse(metrics["expected_modality_hit_at_5"])
        self.assertEqual(
            "DIRECT_GROUNDING_TOP5_MISS",
            metrics["direct_grounding_flags"][0]["flag"],
        )
        self.assertEqual(-0.9, ranked[-1].raw_selection_score)
        self.assertEqual(
            {"transcript": 5},
            metrics["top_5_modality_composition"],
        )

    def test_classification_requires_complete_human_annotations(self) -> None:
        aggregate = {
            "ocr_summary": {"hit_at_5": 0.0},
            "transcript_summary": {"hit_at_5": 1.0},
            "direct_grounding_misses": [{"probe_id": "ocr-probe"}],
            "macro_recall_at_5": 0.5,
        }
        self.assertEqual(
            "INSUFFICIENT_EVIDENCE_TO_CONCLUDE",
            classify_audit(aggregate, annotations_complete=False),
        )
        self.assertEqual(
            "MODALITY_SPECIFIC_RANKING_BIAS",
            classify_audit(aggregate, annotations_complete=True),
        )

    def test_candidate_loader_accepts_public_result_and_preserves_order(self) -> None:
        payload = {
            "outcome": {
                "result": {
                    "evidence": {
                        "g1_exposure_units": [
                            {
                                "unit_id": "transcript-1",
                                "source_type": "transcript",
                                "text": "First transcript unit.",
                                "eligible_for_frozen_g1": True,
                            },
                            {
                                "unit_id": "ocr-1",
                                "source_type": "ocr",
                                "text": "CPAC 2018",
                                "eligible_for_frozen_g1": True,
                            },
                        ]
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public-result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            units = load_candidate_pool(path)

        self.assertEqual(["transcript-1", "ocr-1"], [unit.unit_id for unit in units])
        self.assertTrue(all(unit.selection_score is None for unit in units))

    def test_candidate_loader_rejects_visual_and_formal_data_paths(self) -> None:
        with self.assertRaisesRegex(AuditInputError, "Validation/Test"):
            load_candidate_pool(Path("/tmp/Test/public-result.json"))

        payload = [
            {
                "unit_id": "visual-1",
                "source_type": "visual_observation",
                "text": "A podium is visible.",
                "eligible_for_frozen_g1": False,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public-result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AuditInputError, "visual observations"):
                load_candidate_pool(path)

    def test_manifest_contains_all_controlled_probe_families(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "selector_fidelity_audit"
            / "probe_definitions.json"
        )
        manifest, probes = load_probe_manifest(manifest_path)

        self.assertEqual(10, len(probes))
        self.assertEqual(
            {"OCR", "TRANSCRIPT", "VISUAL_SUPPLEMENTAL", "NONE"},
            {probe.expected_modality for probe in probes},
        )
        self.assertTrue(manifest["scientific_boundary"]["top_k_is_explanation_only"])
        self.assertTrue(
            manifest["scientific_boundary"][
                "prediction_uses_all_valid_unit_veracity_logits"
            ]
        )


if __name__ == "__main__":
    unittest.main()
