import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
    VerificationResult,
)
from services.claim_consistency_gate import (
    CLAIM_VIDEO_MISMATCH_WARNING,
    ClaimConsistencyGate,
    ConsistencyResult,
)
from services.evidence_sufficiency_policy import EvidenceSufficiencyPolicy
from services.production_result import ProductionResultBuilder
from services.video_multimodal_runner import VideoMultimodalRunner


def _unit(unit_id: str, source_type: SourceType, text: str) -> RuntimeUnit:
    return RuntimeUnit(
        unit_id=unit_id,
        source_type=source_type,
        text=text,
        producer="test",
        eligible_for_frozen_g1=source_type is not SourceType.VISUAL_OBSERVATION,
    )


class ClaimConsistencyGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcript = _unit(
            "transcript-1",
            SourceType.TRANSCRIPT,
            "Players dribble a basketball across the indoor court.",
        )
        self.ocr = _unit(
            "ocr-1",
            SourceType.OCR,
            "Home team scoreboard and game clock",
        )
        self.visual = _unit(
            "visual-1",
            SourceType.VISUAL_OBSERVATION,
            "Two teams play basketball beneath a scoreboard.",
        )

    def _runner(self):
        return self._runner_with_units(
            [self.transcript],
            [self.ocr],
            [self.visual],
        )

    @staticmethod
    def _runner_with_units(transcript_units, ocr_units, visual_units):
        text_runner = mock.Mock()
        text_runner.run.return_value = SimpleNamespace(
            g1_exposure_units=list(transcript_units) + list(ocr_units),
            warnings=[],
        )
        visual_runner = mock.Mock()
        visual_runner.run.return_value = SimpleNamespace(
            runtime_units=list(visual_units),
            warnings=[],
        )
        frozen_runner = mock.Mock()
        runner = VideoMultimodalRunner(
            text_runner,
            visual_runner,
            frozen_runner,
        )
        return runner, frozen_runner

    @staticmethod
    def _completed_real(session_id, claim, units):
        return VerificationResult(
            session_id=session_id,
            claim=claim,
            model_verdict=ModelVerdict.REAL,
            display_verdict=DisplayVerdict.REAL,
            evidence_status=EvidenceStatus.SUFFICIENT,
            sample_logits={"fake": 0.1, "real": 0.9},
            probabilities={"fake": 0.2, "real": 0.8},
            all_units=list(units),
            top_k_units=list(units[:1]),
            class_winners={},
            pipeline_stages=[],
            warnings=[],
        )

    def test_unrelated_claim_video_abstains_without_frozen_g1(self) -> None:
        claim = "A cat is sleeping peacefully on a sofa."
        runner, frozen_runner = self._runner()

        result = runner.run("session-mismatch", claim, Path("input.mp4"))

        self.assertIs(
            ClaimConsistencyGate().evaluate(
                claim,
                [self.transcript],
                [self.ocr],
                [self.visual],
            ),
            ConsistencyResult.MISMATCH,
        )
        frozen_runner.run.assert_not_called()
        self.assertIs(result.verification_result.model_verdict, ModelVerdict.NOT_RUN)
        self.assertIs(result.verification_result.display_verdict, DisplayVerdict.NEI)
        self.assertIs(
            result.verification_result.evidence_status,
            EvidenceStatus.INSUFFICIENT,
        )
        self.assertIn(CLAIM_VIDEO_MISMATCH_WARNING, result.warnings)
        self.assertIn(
            CLAIM_VIDEO_MISMATCH_WARNING,
            result.verification_result.warnings,
        )
        assessment = EvidenceSufficiencyPolicy().assess(result)
        self.assertFalse(assessment.model_was_run)
        self.assertEqual(
            assessment.reason_code,
            EvidenceSufficiencyPolicy.CONSISTENCY_MISMATCH_REASON,
        )
        production_result = ProductionResultBuilder().build(result)
        self.assertIs(production_result.model_verdict, ModelVerdict.NOT_RUN)
        self.assertIs(production_result.display_verdict, DisplayVerdict.NEI)
        self.assertEqual((), production_result.sample_logits)
        self.assertEqual((), production_result.probabilities)

    def test_existing_unrelated_claim_examples_remain_nei(self) -> None:
        claims = (
            "This video shows a football match.",
            "A cat is walking outdoors.",
        )
        for index, claim in enumerate(claims):
            with self.subTest(claim=claim):
                runner, frozen_runner = self._runner()

                result = runner.run(
                    f"session-unrelated-{index}", claim, Path("input.mp4")
                )

                frozen_runner.run.assert_not_called()
                self.assertIs(
                    result.verification_result.model_verdict,
                    ModelVerdict.NOT_RUN,
                )
                self.assertIs(
                    result.verification_result.display_verdict,
                    DisplayVerdict.NEI,
                )

    def test_compatible_claim_preserves_frozen_g1_path(self) -> None:
        claim = "Two teams play basketball on an indoor court."
        runner, frozen_runner = self._runner()
        expected = VerificationResult(
            session_id="session-pass",
            claim=claim,
            model_verdict=ModelVerdict.FAKE,
            display_verdict=DisplayVerdict.FAKE,
            evidence_status=EvidenceStatus.SUFFICIENT,
            sample_logits={"fake": 0.9, "real": 0.1},
            probabilities={"fake": 0.8, "real": 0.2},
            all_units=[self.transcript, self.ocr],
            top_k_units=[self.transcript],
            class_winners={},
            pipeline_stages=[],
            warnings=[],
        )
        frozen_runner.run.return_value = expected

        result = runner.run("session-pass", claim, Path("input.mp4"))

        self.assertIs(
            ClaimConsistencyGate().evaluate(
                claim,
                [self.transcript],
                [self.ocr],
                [self.visual],
            ),
            ConsistencyResult.PASS,
        )
        self.assertIs(result.verification_result, expected)
        frozen_runner.run.assert_called_once_with(
            "session-pass",
            claim,
            [self.transcript, self.ocr],
        )
        self.assertIs(result.verification_result.model_verdict, ModelVerdict.FAKE)
        self.assertIs(result.verification_result.display_verdict, DisplayVerdict.FAKE)

    def test_uncertain_claim_preserves_frozen_g1_path(self) -> None:
        claim = "Something may be happening in this video."
        runner, frozen_runner = self._runner()
        expected = VerificationResult(
            session_id="session-unknown",
            claim=claim,
            model_verdict=ModelVerdict.REAL,
            display_verdict=DisplayVerdict.REAL,
            evidence_status=EvidenceStatus.SUFFICIENT,
            sample_logits={"fake": 0.1, "real": 0.9},
            probabilities={"fake": 0.2, "real": 0.8},
            all_units=[self.transcript, self.ocr],
            top_k_units=[self.ocr],
            class_winners={},
            pipeline_stages=[],
            warnings=[],
        )
        frozen_runner.run.return_value = expected

        result = runner.run("session-unknown", claim, Path("input.mp4"))

        self.assertIs(
            ClaimConsistencyGate().evaluate(
                claim,
                [self.transcript],
                [self.ocr],
                [self.visual],
            ),
            ConsistencyResult.UNKNOWN,
        )
        self.assertIs(result.verification_result, expected)
        frozen_runner.run.assert_called_once_with(
            "session-unknown",
            claim,
            [self.transcript, self.ocr],
        )
        self.assertIs(result.verification_result.model_verdict, ModelVerdict.REAL)
        self.assertIs(result.verification_result.display_verdict, DisplayVerdict.REAL)

    def test_unknown_gate_result_for_normal_claim_calls_frozen_g1(self) -> None:
        claim = "A local council approved the proposed transit plan."
        text_runner = mock.Mock()
        text_runner.run.return_value = SimpleNamespace(
            g1_exposure_units=[self.transcript],
            warnings=[],
        )
        visual_runner = mock.Mock()
        visual_runner.run.return_value = SimpleNamespace(
            runtime_units=[],
            warnings=[],
        )
        frozen_runner = mock.Mock()
        runner = VideoMultimodalRunner(
            text_runner,
            visual_runner,
            frozen_runner,
        )
        expected = object()
        frozen_runner.run.return_value = expected

        result = runner.run("session-unknown-normal", claim, Path("input.mp4"))

        self.assertIs(
            ClaimConsistencyGate().evaluate(
                claim,
                [self.transcript],
                [],
                [],
            ),
            ConsistencyResult.UNKNOWN,
        )
        self.assertIs(result.verification_result, expected)
        frozen_runner.run.assert_called_once_with(
            "session-unknown-normal",
            claim,
            [self.transcript],
        )

    def test_explicit_year_mismatch_abstains_before_frozen_g1(self) -> None:
        evidence = _unit(
            "transcript-cpac",
            SourceType.TRANSCRIPT,
            "CPAC 2018",
        )
        for year in (2019, 2026):
            with self.subTest(year=year):
                claim = f"This video was recorded in {year}."
                runner, frozen_runner = self._runner_with_units(
                    [evidence], [], []
                )

                result = runner.run(
                    f"session-year-{year}", claim, Path("input.mp4")
                )

                frozen_runner.run.assert_not_called()
                self.assertIs(
                    result.verification_result.model_verdict,
                    ModelVerdict.NOT_RUN,
                )
                self.assertIs(
                    result.verification_result.display_verdict,
                    DisplayVerdict.NEI,
                )
                self.assertIs(
                    result.verification_result.evidence_status,
                    EvidenceStatus.INSUFFICIENT,
                )

    def test_conflicting_named_person_abstains_before_frozen_g1(self) -> None:
        evidence = _unit(
            "transcript-trump",
            SourceType.TRANSCRIPT,
            "Donald Trump speaking at CPAC 2018",
        )
        claim = "This video shows Barack Obama speaking at CPAC 2018."
        runner, frozen_runner = self._runner_with_units([evidence], [], [])

        result = runner.run("session-entity-mismatch", claim, Path("input.mp4"))

        frozen_runner.run.assert_not_called()
        self.assertIs(result.verification_result.model_verdict, ModelVerdict.NOT_RUN)
        self.assertIs(result.verification_result.display_verdict, DisplayVerdict.NEI)
        self.assertIn(CLAIM_VIDEO_MISMATCH_WARNING, result.warnings)

    def test_matching_named_person_preserves_real_frozen_g1_result(self) -> None:
        evidence = _unit(
            "transcript-trump",
            SourceType.TRANSCRIPT,
            "Donald Trump speaking at CPAC 2018",
        )
        claim = "This video shows Donald Trump speaking at CPAC 2018."
        runner, frozen_runner = self._runner_with_units([evidence], [], [])
        expected = self._completed_real(
            "session-entity-match", claim, [evidence]
        )
        frozen_runner.run.return_value = expected

        result = runner.run("session-entity-match", claim, Path("input.mp4"))

        frozen_runner.run.assert_called_once_with(
            "session-entity-match", claim, [evidence]
        )
        self.assertIs(result.verification_result, expected)
        self.assertIs(result.verification_result.model_verdict, ModelVerdict.REAL)

    def test_person_title_does_not_create_false_entity_mismatch(self) -> None:
        evidence = _unit(
            "transcript-president-trump",
            SourceType.TRANSCRIPT,
            "President Donald Trump speaking at CPAC 2018",
        )
        claim = "This video shows Donald Trump speaking at CPAC 2018."

        result = ClaimConsistencyGate().evaluate(claim, [evidence], [], [])

        self.assertIs(result, ConsistencyResult.PASS)

    def test_unsupported_named_person_abstains_even_when_event_matches(self) -> None:
        evidence = _unit(
            "ocr-cpac-only",
            SourceType.OCR,
            "CPAC 2018",
        )
        claim = "This video shows Donald Trump speaking at CPAC 2018."
        runner, frozen_runner = self._runner_with_units([], [evidence], [])

        result = runner.run("session-entity-unsupported", claim, Path("input.mp4"))

        frozen_runner.run.assert_not_called()
        self.assertIs(result.verification_result.model_verdict, ModelVerdict.NOT_RUN)
        self.assertIs(result.verification_result.display_verdict, DisplayVerdict.NEI)

    def test_matching_event_and_year_preserve_real_frozen_g1_result(self) -> None:
        evidence = _unit(
            "ocr-cpac",
            SourceType.OCR,
            "CPAC 2018",
        )
        claim = "This video contains information about CPAC 2018."
        runner, frozen_runner = self._runner_with_units([], [evidence], [])
        expected = self._completed_real("session-cpac-match", claim, [evidence])
        frozen_runner.run.return_value = expected

        result = runner.run("session-cpac-match", claim, Path("input.mp4"))

        frozen_runner.run.assert_called_once_with(
            "session-cpac-match", claim, [evidence]
        )
        self.assertIs(result.verification_result, expected)

    def test_claim_without_explicit_year_does_not_trigger_temporal_mismatch(self) -> None:
        evidence = _unit(
            "ocr-cpac-year",
            SourceType.OCR,
            "CPAC 2018",
        )

        result = ClaimConsistencyGate().evaluate(
            "This video is about CPAC.", [], [evidence], []
        )

        self.assertIs(result, ConsistencyResult.PASS)


if __name__ == "__main__":
    unittest.main()
