"""Deterministic end-to-end mock runtime."""

from pathlib import Path
from typing import Dict, Optional

from schemas import StageName, VerificationRequest, VerificationResult
from services.cache_manager import CacheManager, safe_target, write_json
from services.logger import RuntimeLogger
from services.mock_models import MockASR, MockG1, MockOCR, MockVLM, MockVisualRetriever
from services.model_registry import ModelRegistry
from services.session_manager import SessionManager

from .pipeline_context import PipelineContext, RuntimeConfig
from .stages import build_runtime_unit_pool, run_asr, run_ocr, run_visual_retrieval, run_vlm


class RuntimeOrchestrator:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.cache = CacheManager(config.cache_root)
        self.logger = RuntimeLogger(config.output_root)
        self.sessions = SessionManager(self.cache)
        self.registry = ModelRegistry()
        self.asr = MockASR()
        self.ocr = MockOCR()
        self.visual_retriever = MockVisualRetriever()
        self.vlm = MockVLM()
        self.g1 = MockG1(config.max_units, config.top_k)
        self.last_result_path: Optional[Path] = None

    def _complete(self, context: PipelineContext, name: StageName, detail: str) -> None:
        context.complete_stage(name, detail)
        pending = getattr(context, "_pending_stage_records", [])
        pending.append({"detail": detail, "stage": name.value})
        if context.session_id:
            for record in pending:
                self.logger.log(context.session_id, "stage_completed", record)
            pending.clear()
        context._pending_stage_records = pending

    def run(self, request: VerificationRequest) -> VerificationResult:
        context = PipelineContext(request, self.config)

        context.start_stage(StageName.REQUEST)
        self._complete(context, StageName.REQUEST, "request accepted")

        context.start_stage(StageName.SESSION)
        context.session_id = self.sessions.create(request)
        self._complete(context, StageName.SESSION, "deterministic session created")

        context.start_stage(StageName.ASR)
        transcript_units = run_asr(request, self.asr)
        self._complete(context, StageName.ASR, f"{len(transcript_units)} transcript units")

        context.start_stage(StageName.OCR)
        ocr_units = run_ocr(request, self.ocr)
        self._complete(context, StageName.OCR, f"{len(ocr_units)} OCR units")

        context.start_stage(StageName.VISUAL_RETRIEVAL)
        candidates = run_visual_retrieval(request, self.visual_retriever)
        self._complete(context, StageName.VISUAL_RETRIEVAL, f"{len(candidates)} visual candidates")

        context.start_stage(StageName.VLM)
        visual_units = run_vlm(candidates, self.vlm)
        self._complete(context, StageName.VLM, f"{len(visual_units)} visual observations")

        context.start_stage(StageName.UNIT_POOL)
        context.units = build_runtime_unit_pool(transcript_units, ocr_units, visual_units)
        self.cache.put_json("units", context.session_id, {"units": [unit.to_dict() for unit in context.units]})
        self._complete(context, StageName.UNIT_POOL, f"{len(context.units)} total units")

        context.start_stage(StageName.MOCK_G1)
        mock_output = self.g1.evaluate(request.claim, context.units)
        evaluated_count = sum(unit.logits is not None for unit in context.units)
        self._complete(context, StageName.MOCK_G1, f"{evaluated_count} eligible units evaluated")

        context.start_stage(StageName.RESULT)
        self._complete(context, StageName.RESULT, "result assembled")

        result = VerificationResult(
            session_id=context.session_id,
            claim=request.claim,
            model_verdict=mock_output.model_verdict,
            display_verdict=mock_output.display_verdict,
            evidence_status=mock_output.evidence_status,
            sample_logits=mock_output.sample_logits,
            probabilities=mock_output.probabilities,
            all_units=context.units,
            top_k_units=mock_output.top_k_units,
            class_winners=mock_output.class_winners,
            pipeline_stages=context.stages,
            warnings=context.warnings,
            checkpoint_sha256=None,
            runtime_ms=0.0,
        )
        target = safe_target(self.config.output_root, "results", f"{context.session_id}.json")
        self.last_result_path = write_json(target, result.to_dict())
        self.logger.log(context.session_id, "result_created", {"path": str(self.last_result_path)})
        return result
