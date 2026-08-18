"""Pure stage adapters used by the orchestrator."""

from typing import Dict, List

from schemas import RuntimeUnit, VerificationRequest
from services.mock_models import MockASR, MockOCR, MockVLM, MockVisualRetriever


def run_asr(request: VerificationRequest, model: MockASR) -> List[RuntimeUnit]:
    return model.transcribe(request)


def run_ocr(request: VerificationRequest, model: MockOCR) -> List[RuntimeUnit]:
    return model.extract(request)


def run_visual_retrieval(request: VerificationRequest, model: MockVisualRetriever) -> List[Dict[str, object]]:
    return model.retrieve(request)


def run_vlm(candidates: List[Dict[str, object]], model: MockVLM) -> List[RuntimeUnit]:
    return model.observe(candidates)


def build_runtime_unit_pool(*groups: List[RuntimeUnit]) -> List[RuntimeUnit]:
    return [unit for group in groups for unit in group]
