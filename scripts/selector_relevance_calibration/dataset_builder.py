"""Build score-blind Train-only direct-relevance calibration records.

This module constructs data only.  It never loads Frozen G1, reads veracity
labels, inspects selector outputs, or executes model inference.  Candidate
exposure is delegated to the real Phase4A ``normalize_request`` function on
DICC, or to an explicitly injected fixture adapter in unit tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from scripts.selector_fidelity_audit.cross_case import (
    canonicalize_underlying_case_id,
)


SCHEMA_VERSION = 1
IMPLEMENTATION_REVISION = "step2.6r-1a-v1"
CALIBRATED_SELECTOR_ID = "G1-RelevanceSelector-Cal-v1"
CPAC_HELDOUT_ID = "GroundLie360:13025004"
AUTHORITATIVE_TRAIN_SHA256 = (
    "e807535556441434df0ef53a37921c0bdac5e27215ed045104ac08f38275e406"
)
EXPECTED_STEP25B_HELDOUT_IDS = frozenset(
    {
        "TRUE-3MFact:10145403",
        "TRUE-3MFact:10258205",
        "TRUE-3MFact:10372904",
        "TRUE-3MFact:10455808",
        "TRUE-3MFact:10865013",
    }
)
FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "label",
        "veracity_label",
        "selection_score",
        "selection_scores",
        "selection_probability",
        "selection_probabilities",
        "veracity_logits",
        "sample_logits",
        "probabilities",
        "prediction",
        "prediction_id",
        "top_k",
        "topk",
        "top_k_selection_units",
        "top_k_membership",
        "rank",
    }
)
_SPLIT_PATH_COMPONENTS = {"validation", "test"}
_PROVENANCE_FIELDS = (
    "unit_id",
    "snippet_id",
    "snippet_path",
    "source",
    "source_id",
    "source_path",
    "evidence_refs",
    "frame_ids",
    "grounding",
    "grounding_provenance",
    "provenance",
)
_AMBIGUOUS_PROVENANCE_MARKERS = (
    ":test:",
    ":validation:",
    "/test/",
    "/validation/",
)
_URL_RE = re.compile(r"^(?:(?:https?://|www\.)\S+)(?:\s+(?:https?://|www\.)\S+)*$", re.I)
_HANDLE_RE = re.compile(r"^@[^\s@]+$")
_NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)*$")
_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
_TEMPERATURE_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:°)?[CFcf]$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DatasetBuildError(ValueError):
    """Raised when a scientific or provenance boundary cannot be preserved."""


class FrozenExposureUnavailableError(DatasetBuildError):
    """Raised when the real Phase4A exposure implementation is unavailable."""


@dataclass(frozen=True)
class ExposureResult:
    """Exact candidate exposure returned by an injected Frozen policy."""

    candidate_units: Tuple[Mapping[str, Any], ...]
    source_candidate_count: int
    truncated_count: int
    dropped_unsupported_count: int

    def __post_init__(self) -> None:
        counts = (
            self.source_candidate_count,
            self.truncated_count,
            self.dropped_unsupported_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise DatasetBuildError("Frozen exposure accounting must be nonnegative integers")
        if (
            len(self.candidate_units)
            + self.truncated_count
            + self.dropped_unsupported_count
            != self.source_candidate_count
        ):
            raise DatasetBuildError("Frozen exposure accounting is inconsistent")


class FrozenExposureAdapter(Protocol):
    """Minimal injectable boundary around Phase4A request normalization."""

    def normalize(self, request: Mapping[str, Any]) -> ExposureResult:
        ...


@dataclass(frozen=True)
class TrainLock:
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class Candidate:
    unit_id: str
    unit_type: str
    modality: str
    text: str
    normalized_text: str
    signature: str
    anchor_modality: Optional[str]
    exposure_index: int

    def public_record(self, relevance_target: int) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "modality": self.modality,
            "text": self.text,
            "relevance_target": relevance_target,
        }


@dataclass(frozen=True)
class SourceCase:
    source_dataset: str
    source_case_id: str
    canonical_underlying_case_id: str
    source_row_index: int
    request: Mapping[str, Any]
    candidates: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class EligibleCase:
    source_dataset: str
    source_case_id: str
    canonical_underlying_case_id: str
    source_row_index: int
    source_candidate_count: int
    truncated_count: int
    dropped_unsupported_count: int
    candidates: Tuple[Candidate, ...]
    transcript_anchors: Tuple[Candidate, Candidate]
    ocr_anchors: Tuple[Candidate, Candidate]


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    build_report: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.casefold()):
        raise DatasetBuildError(f"{field} must be a lowercase SHA-256 digest")
    return value.casefold()


def _reject_formal_path(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(part.casefold() in _SPLIT_PATH_COMPONENTS for part in resolved.parts):
        raise DatasetBuildError(f"{field} must not reference formal Validation/Test")
    return resolved


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetBuildError(f"{field} is missing") from exc
    except json.JSONDecodeError as exc:
        raise DatasetBuildError(f"{field} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise DatasetBuildError(f"{field} must contain a JSON object")
    return payload


def verify_train_lock(
    project_root: Path,
    report_path: Path,
    expected_sha256: Optional[str] = AUTHORITATIVE_TRAIN_SHA256,
) -> TrainLock:
    project_root = Path(project_root).expanduser().resolve()
    report_path = _reject_formal_path(report_path, "Phase3A Train-lock report")
    report = _read_json(report_path, "Phase3A Train-lock report")
    section = report.get("train_lock")
    if section is None:
        section = report
    if not isinstance(section, Mapping):
        raise DatasetBuildError("Phase3A Train-lock metadata is required")
    status = section.get("status")
    if status is None:
        status = report.get("status")
    if status != "PASS":
        raise DatasetBuildError("Phase3A Train-lock status must be PASS")
    source = section.get("source")
    if source is None:
        source = report.get("source")
    if not isinstance(source, Mapping):
        raise DatasetBuildError("Phase3A Train-lock source metadata is required")
    raw_path = source.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise DatasetBuildError("Phase3A Train-lock source.path is required")
    source_path = Path(raw_path).expanduser()
    if not source_path.is_absolute():
        source_path = project_root / source_path
    source_path = _reject_formal_path(source_path, "authoritative Train source")
    if not source_path.is_file():
        raise DatasetBuildError("authoritative Train source is missing")
    declared_sha = _require_sha256(source.get("sha256"), "Train-lock source.sha256")
    actual_sha = sha256_file(source_path)
    if actual_sha != declared_sha:
        raise DatasetBuildError("authoritative Train source SHA-256 mismatch")
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, "expected Train source SHA-256")
        if declared_sha != expected:
            raise DatasetBuildError("Train-lock source does not match the authoritative Train SHA-256")
    return TrainLock(source_path=source_path, source_sha256=actual_sha)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(
        str.maketrans(
            {
                "‘": "'",
                "’": "'",
                "‚": "'",
                "‛": "'",
                "“": '"',
                "”": '"',
                "„": '"',
                "‟": '"',
            }
        )
    )
    return " ".join(normalized.split()).strip()


def lexical_signature(text: str) -> str:
    return normalize_text(text).casefold()


def _non_whitespace_alphanumeric_fraction(text: str) -> float:
    characters = [character for character in text if not character.isspace()]
    if not characters:
        return 0.0
    return sum(character.isalnum() for character in characters) / len(characters)


def _tokens(text: str) -> List[str]:
    return text.split()


def _alphabetic_token_count(tokens: Sequence[str]) -> int:
    return sum(any(character.isalpha() for character in token) for token in tokens)


def transcript_quality_reason(text: Any) -> Optional[str]:
    normalized = normalize_text(text)
    tokens = _tokens(normalized)
    if len(normalized) < 20:
        return "transcript_character_length_below_20"
    if len(normalized) > 240:
        return "transcript_character_length_above_240"
    if _URL_RE.fullmatch(normalized):
        return "transcript_url_only"
    if tokens and all(_HANDLE_RE.fullmatch(token) for token in tokens):
        return "transcript_handle_only"
    if len(tokens) < 4:
        return "transcript_token_count_below_4"
    if len(tokens) > 40:
        return "transcript_token_count_above_40"
    if _alphabetic_token_count(tokens) < 3:
        return "transcript_alphabetic_token_count_below_3"
    if _non_whitespace_alphanumeric_fraction(normalized) < 0.70:
        return "transcript_alphanumeric_fraction_below_0_70"
    return None


def _noisy_ocr_token(token: str) -> bool:
    return bool(
        _NUMBER_RE.fullmatch(token)
        or _TIMESTAMP_RE.fullmatch(token)
        or _TEMPERATURE_RE.fullmatch(token)
        or (len(token) == 1 and not token.isalnum())
    )


def ocr_quality_reason(text: Any) -> Optional[str]:
    normalized = normalize_text(text)
    tokens = _tokens(normalized)
    if len(normalized) < 20:
        return "ocr_character_length_below_20"
    if len(normalized) > 240:
        return "ocr_character_length_above_240"
    if _URL_RE.fullmatch(normalized):
        return "ocr_url_only"
    if len(tokens) < 3:
        return "ocr_token_count_below_3"
    if len(tokens) > 40:
        return "ocr_token_count_above_40"
    if _alphabetic_token_count(tokens) < 2:
        return "ocr_alphabetic_token_count_below_2"
    if sum(character.isalpha() for character in normalized) < 10:
        return "ocr_alphabetic_character_count_below_10"
    if _non_whitespace_alphanumeric_fraction(normalized) < 0.65:
        return "ocr_alphanumeric_fraction_below_0_65"
    if tokens and sum(_noisy_ocr_token(token) for token in tokens) / len(tokens) > 0.50:
        return "ocr_numeric_timestamp_noise_above_0_50"
    return None


def candidate_anchor_modality(unit_type: Any, modality: Any) -> Optional[str]:
    normalized_type = normalize_text(unit_type).casefold()
    normalized_modality = normalize_text(modality).casefold()
    if normalized_type == "transcript":
        return "TRANSCRIPT"
    if normalized_type == "ocr" or normalized_modality == "ocr":
        return "OCR"
    return None


def _ambiguous_string(value: str) -> bool:
    normalized = value.replace("\\", "/").casefold()
    return any(marker in normalized for marker in _AMBIGUOUS_PROVENANCE_MARKERS)


def _provenance_value_is_ambiguous(value: Any, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, str):
        return _ambiguous_string(value)
    if isinstance(value, (list, tuple)):
        return any(_provenance_value_is_ambiguous(item, depth + 1) for item in value)
    if isinstance(value, Mapping):
        return any(
            _provenance_value_is_ambiguous(value.get(field), depth + 1)
            for field in _PROVENANCE_FIELDS
        )
    return False


def row_has_ambiguous_provenance(candidates: Sequence[Mapping[str, Any]]) -> bool:
    for candidate in candidates:
        if any(
            _provenance_value_is_ambiguous(candidate.get(field))
            for field in _PROVENANCE_FIELDS
        ):
            return True
    return False


def _candidate_request_record(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "unit_id": candidate.get("unit_id"),
        "unit_type": candidate.get("unit_type"),
        "modality": candidate.get("modality"),
        "text": candidate.get("text"),
    }


def _source_case_from_row(row: Mapping[str, Any], row_index: int) -> SourceCase:
    dataset = row.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        dataset = row.get("source_dataset")
    case_id = row.get("case_id")
    if not isinstance(case_id, (str, int)) or not str(case_id).strip():
        case_id = row.get("sample_id")
    if not isinstance(case_id, (str, int)) or not str(case_id).strip():
        case_id = row.get("id")
    claim = row.get("claim")
    raw_candidates = row.get("candidate_units")
    if not isinstance(dataset, str) or not dataset.strip():
        raise DatasetBuildError("Train row dataset is missing")
    if not isinstance(case_id, (str, int)) or not str(case_id).strip():
        raise DatasetBuildError("Train row case identity is missing")
    if not isinstance(claim, str) or not claim.strip():
        raise DatasetBuildError("Train row claim is missing")
    if not isinstance(raw_candidates, list) or not all(
        isinstance(candidate, Mapping) for candidate in raw_candidates
    ):
        raise DatasetBuildError("Train row candidate_units must be a list of objects")
    candidates = tuple(raw_candidates)
    source_dataset = dataset.strip()
    source_case_id = str(case_id).strip()
    request = {
        "dataset": source_dataset,
        "case_id": source_case_id,
        "claim": claim.strip(),
        "candidate_units": [_candidate_request_record(candidate) for candidate in candidates],
    }
    return SourceCase(
        source_dataset=source_dataset,
        source_case_id=source_case_id,
        canonical_underlying_case_id=canonicalize_underlying_case_id(
            source_dataset, source_case_id
        ),
        source_row_index=row_index,
        request=request,
        candidates=candidates,
    )


def iter_jsonl_rows(path: Path) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(f"malformed Train JSONL row {row_index}") from exc
            if not isinstance(row, dict):
                raise DatasetBuildError(f"Train JSONL row {row_index} must be an object")
            yield row_index, row


def load_step25b_heldout_ids(path: Path) -> Tuple[str, ...]:
    path = _reject_formal_path(path, "Step 2.5B selected manifest")
    manifest = _read_json(path, "Step 2.5B selected manifest")
    selected = manifest.get("selected_cases")
    if not isinstance(selected, list):
        raise DatasetBuildError("Step 2.5B selected manifest is missing selected_cases")
    identities = []
    for item in selected:
        if not isinstance(item, Mapping):
            raise DatasetBuildError("Step 2.5B selected case must be an object")
        identity = item.get("canonical_underlying_case_id")
        if not isinstance(identity, str) or not identity.strip():
            raise DatasetBuildError("Step 2.5B selected case lacks canonical identity")
        identities.append(identity.strip())
    if set(identities) != EXPECTED_STEP25B_HELDOUT_IDS or len(identities) != len(
        EXPECTED_STEP25B_HELDOUT_IDS
    ):
        raise DatasetBuildError("Step 2.5B manifest must contain exactly the expected five identities")
    return tuple(sorted(identities))


def _candidate_from_exposed(record: Mapping[str, Any], exposure_index: int) -> Candidate:
    unit_id = record.get("unit_id")
    unit_type = record.get("unit_type")
    modality = record.get("modality")
    text = record.get("text")
    if not isinstance(unit_id, str) or not unit_id.strip():
        raise DatasetBuildError("model-exposed candidate unit_id must be non-blank")
    if not isinstance(unit_type, str) or not unit_type.strip():
        raise DatasetBuildError("model-exposed candidate unit_type must be non-blank")
    if not isinstance(modality, str) or not modality.strip():
        raise DatasetBuildError("model-exposed candidate modality must be non-blank")
    if not isinstance(text, str) or not text.strip():
        raise DatasetBuildError("model-exposed candidate text must be non-blank")
    normalized = normalize_text(text)
    return Candidate(
        unit_id=unit_id.strip(),
        unit_type=unit_type.strip(),
        modality=modality.strip(),
        text=text,
        normalized_text=normalized,
        signature=normalized.casefold(),
        anchor_modality=candidate_anchor_modality(unit_type, modality),
        exposure_index=exposure_index,
    )


def _quality_reason(candidate: Candidate) -> Optional[str]:
    if candidate.anchor_modality == "TRANSCRIPT":
        return transcript_quality_reason(candidate.normalized_text)
    if candidate.anchor_modality == "OCR":
        return ocr_quality_reason(candidate.normalized_text)
    return "unsupported_anchor_modality"


def _select_anchor_groups(
    canonical_id: str,
    modality: str,
    candidates: Sequence[Candidate],
) -> Tuple[Tuple[Candidate, ...], Counter]:
    exclusions: Counter = Counter()
    groups: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.anchor_modality != modality:
            continue
        reason = _quality_reason(candidate)
        if reason is not None:
            exclusions[reason] += 1
            continue
        groups[candidate.signature].append(candidate)
    ranked_groups = []
    for signature, group in groups.items():
        representative = min(group, key=lambda item: (item.unit_id, item.exposure_index))
        stable_key = hashlib.sha256(
            (
                canonical_id
                + "|"
                + modality
                + "|"
                + signature
                + "|"
                + representative.unit_id
            ).encode("utf-8")
        ).hexdigest()
        ranked_groups.append((stable_key, representative.unit_id, representative))
    ranked_groups.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked_groups[:2]), exclusions


def _strict_positive(anchor: Candidate, candidate: Candidate) -> bool:
    if anchor.anchor_modality != candidate.anchor_modality:
        return False
    left = anchor.signature
    right = candidate.signature
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return (
        len(shorter) >= 20
        and shorter in longer
        and len(shorter) / len(longer) >= 0.80
    )


def _calibration_example_id(case: EligibleCase, modality: str, anchor: Candidate) -> str:
    digest = hashlib.sha256(
        (
            CALIBRATED_SELECTOR_ID
            + "|"
            + case.canonical_underlying_case_id
            + "|"
            + modality
            + "|"
            + anchor.unit_id
        ).encode("utf-8")
    ).hexdigest()
    return "direct-relevance-" + digest[:24]


def make_calibration_example(
    case: EligibleCase,
    split: str,
    modality: str,
    anchor: Candidate,
    train_sha256: str,
    phase4a_config_sha256: str,
) -> Dict[str, Any]:
    if split not in {"train", "dev"}:
        raise DatasetBuildError("calibration split must be train or dev")
    if modality == "OCR":
        claim = f'The on-screen text reads "{anchor.normalized_text}".'
    elif modality == "TRANSCRIPT":
        claim = f'The speaker says "{anchor.normalized_text}".'
    else:
        raise DatasetBuildError("expected modality must be OCR or TRANSCRIPT")
    targets = [int(_strict_positive(anchor, candidate)) for candidate in case.candidates]
    positive_ids = [
        candidate.unit_id
        for candidate, target in zip(case.candidates, targets)
        if target == 1
    ]
    if anchor.unit_id not in positive_ids or not positive_ids:
        raise DatasetBuildError("calibration example has no representative positive")
    example = {
        "schema_version": SCHEMA_VERSION,
        "calibration_example_id": _calibration_example_id(case, modality, anchor),
        "source_dataset": case.source_dataset,
        "source_case_id": case.source_case_id,
        "canonical_underlying_case_id": case.canonical_underlying_case_id,
        "calibration_split": split,
        "expected_modality": modality,
        "claim": claim,
        "anchor_unit_id": anchor.unit_id,
        "anchor_text": anchor.normalized_text,
        "positive_unit_ids": positive_ids,
        "model_exposed_candidate_count": len(case.candidates),
        "candidate_units": [
            candidate.public_record(target)
            for candidate, target in zip(case.candidates, targets)
        ],
        "source_provenance": {
            "train_variant_sha256": train_sha256,
            "phase4a_config_sha256": phase4a_config_sha256,
            "source_row_index": case.source_row_index,
        },
    }
    assert_no_forbidden_output_fields(example)
    return example


def assert_no_forbidden_output_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_OUTPUT_KEYS:
                raise DatasetBuildError(f"forbidden selector/veracity field leaked: {key}")
            assert_no_forbidden_output_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_forbidden_output_fields(item)


def assign_case_disjoint_splits(cases: Sequence[EligibleCase]) -> Dict[str, str]:
    by_dataset: Dict[str, List[EligibleCase]] = defaultdict(list)
    for case in cases:
        by_dataset[case.source_dataset].append(case)
    assignments: Dict[str, str] = {}
    for dataset in sorted(by_dataset, key=str.casefold):
        ranked = sorted(
            by_dataset[dataset],
            key=lambda case: (
                hashlib.sha256(
                    (
                        CALIBRATED_SELECTOR_ID
                        + "|"
                        + case.canonical_underlying_case_id
                    ).encode("utf-8")
                ).hexdigest(),
                case.canonical_underlying_case_id,
            ),
        )
        for index, case in enumerate(ranked):
            assignments[case.canonical_underlying_case_id] = (
                "dev" if index % 5 == 4 else "train"
            )
    return assignments


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _write_bytes(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    return _write_bytes(path, _json_bytes(payload))


def _write_sidecar(path: Path, digest: str) -> None:
    path.write_text(digest + "\n", encoding="utf-8")


def _manifest_case(case: EligibleCase) -> Dict[str, Any]:
    return {
        "source_dataset": case.source_dataset,
        "source_case_id": case.source_case_id,
        "canonical_underlying_case_id": case.canonical_underlying_case_id,
        "source_row_index": case.source_row_index,
        "source_candidate_pool_count": case.source_candidate_count,
        "model_exposed_candidate_pool_count": len(case.candidates),
        "truncated_count": case.truncated_count,
        "dropped_unsupported_count": case.dropped_unsupported_count,
        "candidate_unit_ids_in_exposure_order": [
            candidate.unit_id for candidate in case.candidates
        ],
        "selected_transcript_anchor_unit_ids": [
            anchor.unit_id for anchor in case.transcript_anchors
        ],
        "selected_ocr_anchor_unit_ids": [anchor.unit_id for anchor in case.ocr_anchors],
    }


def _dataset_card(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# G1-RelevanceSelector-Cal-v1 Train-only Dataset",
            "",
            "## Purpose",
            "",
            "Train-only direct-relevance calibration data for a future explanation selector.",
            "",
            "## Scientific rationale",
            "",
            "Step 2.5B found general selector relevance instability rather than a stable OCR-specific bias. The original Frozen G1 selector was optimized using soft targets distilled from unit-level veracity margins, which is not identical to direct claim-unit grounding relevance.",
            "",
            "The new target is deterministic direct lexical grounding for balanced OCR and Transcript anchor claims. This dataset is not for veracity-core training.",
            "",
            "## Boundaries",
            "",
            "- Step 2.5B five cases and the CPAC strict case are held out.",
            "- Formal Validation was not used.",
            "- Formal Test was not used.",
            "- No source-specific inference bonus or modality quota is introduced.",
            "- No selector outputs, veracity labels, model, or checkpoint were used.",
            f"- Eligible cases: {report['eligible_case_count']}.",
            f"- Calibration Train examples: {report['calibration_train_example_count']}.",
            f"- Calibration Dev examples: {report['calibration_dev_example_count']}.",
            "",
        ]
    )


def _phase4a_count(normalized: Mapping[str, Any], names: Sequence[str]) -> Optional[int]:
    for name in names:
        value = normalized.get(name)
        if type(value) is int and value >= 0:
            return value
    return None


class Phase4ANormalizationExposureAdapter:
    """DICC-only loader for the real pure Phase4A normalization function."""

    def __init__(
        self,
        normalize_request: Any,
        config: Mapping[str, Any],
    ) -> None:
        if not callable(normalize_request):
            raise FrozenExposureUnavailableError("Phase4A normalize_request is unavailable")
        self._normalize_request = normalize_request
        self._config = config
        self._invocation = self._resolve_invocation(normalize_request)

    @staticmethod
    def _resolve_invocation(function: Any) -> str:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError) as exc:
            raise FrozenExposureUnavailableError(
                "Phase4A normalize_request signature cannot be inspected"
            ) from exc
        probes = (
            ("config_keyword", (({},), {"config": {}})),
            ("request_keywords", ((), {"request": {}, "config": {}})),
            ("row_keywords", ((), {"row": {}, "config": {}})),
            ("record_keywords", ((), {"record": {}, "config": {}})),
            ("request_only", (({},), {})),
        )
        for name, (args, kwargs) in probes:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            return name
        raise FrozenExposureUnavailableError(
            "Phase4A normalize_request must accept a request without model execution"
        )

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        phase4a_config_path: Path,
    ) -> "Phase4ANormalizationExposureAdapter":
        project_root = Path(project_root).expanduser().resolve()
        config_path = _reject_formal_path(phase4a_config_path, "Phase4A config")
        config = _read_json(config_path, "Phase4A config")
        engine_path = project_root / "MDU" / "scripts" / "clip12_phase4a_inference_handoff" / "clip12p4a_engine.py"
        if not engine_path.is_file():
            raise FrozenExposureUnavailableError(
                "actual Phase4A normalize_request implementation is missing"
            )
        module_name = "_selector_relevance_phase4a_engine"
        spec = importlib.util.spec_from_file_location(module_name, engine_path)
        if spec is None or spec.loader is None:
            raise FrozenExposureUnavailableError("cannot load Phase4A normalization module")
        module = importlib.util.module_from_spec(spec)
        desired_entries = [str(project_root), str(engine_path.parent)]
        inserted = []
        for entry in reversed(desired_entries):
            if entry not in sys.path:
                sys.path.insert(0, entry)
                inserted.append(entry)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise FrozenExposureUnavailableError(
                "cannot import actual Phase4A normalization module"
            ) from exc
        finally:
            sys.modules.pop(module_name, None)
            for entry in inserted:
                if entry in sys.path:
                    sys.path.remove(entry)
        function = getattr(module, "normalize_request", None)
        return cls(function, config)

    def _call(self, request: Mapping[str, Any]) -> Any:
        if self._invocation == "config_keyword":
            return self._normalize_request(request, config=self._config)
        if self._invocation == "request_keywords":
            return self._normalize_request(request=request, config=self._config)
        if self._invocation == "row_keywords":
            return self._normalize_request(row=request, config=self._config)
        if self._invocation == "record_keywords":
            return self._normalize_request(record=request, config=self._config)
        return self._normalize_request(request)

    def normalize(self, request: Mapping[str, Any]) -> ExposureResult:
        try:
            normalized = self._call(request)
        except TypeError as exc:
            raise FrozenExposureUnavailableError(
                "actual Phase4A normalize_request call is incompatible"
            ) from exc
        if not isinstance(normalized, Mapping):
            raise DatasetBuildError("Phase4A normalize_request must return an object")
        candidates = normalized.get("candidate_units")
        if not isinstance(candidates, list) or not all(
            isinstance(item, Mapping) for item in candidates
        ):
            raise DatasetBuildError("Phase4A normalized candidate_units are invalid")
        source_count = len(request.get("candidate_units", []))
        truncated = _phase4a_count(
            normalized,
            ("truncated_unit_count", "truncated_count"),
        )
        if truncated is None:
            truncated = max(0, source_count - 24)
        dropped = _phase4a_count(
            normalized,
            (
                "dropped_unsupported_unit_count",
                "dropped_unsupported_count",
                "dropped_unit_count",
            ),
        )
        if dropped is None:
            dropped = source_count - len(candidates) - truncated
        if dropped < 0:
            raise DatasetBuildError("Phase4A normalized exposure accounting is invalid")
        return ExposureResult(
            candidate_units=tuple(candidates),
            source_candidate_count=source_count,
            truncated_count=truncated,
            dropped_unsupported_count=dropped,
        )


def build_calibration_dataset(
    *,
    project_root: Path,
    phase3a_train_lock_report: Path,
    phase4a_config_path: Path,
    step25b_selected_manifest: Path,
    heldout_cases: Sequence[str],
    output_dir: Path,
    exposure_adapter: Optional[FrozenExposureAdapter] = None,
    expected_train_sha256: Optional[str] = AUTHORITATIVE_TRAIN_SHA256,
) -> BuildResult:
    project_root = Path(project_root).expanduser().resolve()
    phase4a_config_path = _reject_formal_path(phase4a_config_path, "Phase4A config")
    if not phase4a_config_path.is_file():
        raise DatasetBuildError("Phase4A config is missing")
    phase4a_config_sha256 = sha256_file(phase4a_config_path)
    train_lock = verify_train_lock(
        project_root,
        phase3a_train_lock_report,
        expected_sha256=expected_train_sha256,
    )
    selected_heldout = load_step25b_heldout_ids(step25b_selected_manifest)
    additional_heldout = tuple(
        sorted(
            {
                normalize_text(identity)
                for identity in heldout_cases
                if normalize_text(identity)
            }
        )
    )
    heldout = frozenset(selected_heldout) | frozenset(additional_heldout)
    heldout_casefold = {identity.casefold() for identity in heldout}
    if not heldout_cases:
        raise DatasetBuildError("at least one explicit held-out case is required")
    if CPAC_HELDOUT_ID.casefold() not in {
        identity.casefold() for identity in additional_heldout
    }:
        raise DatasetBuildError("the CPAC strict-audit identity must be explicitly held out")
    if exposure_adapter is None:
        exposure_adapter = Phase4ANormalizationExposureAdapter.from_project_root(
            project_root, phase4a_config_path
        )
    output_dir = Path(output_dir).expanduser().resolve()
    if train_lock.source_path.parent == output_dir or train_lock.source_path.parent in output_dir.parents:
        raise DatasetBuildError("output directory must not be inside the Train source directory")

    source_row_count = 0
    malformed_source_row_count = 0
    ambiguous_provenance_exclusion_count = 0
    heldout_exclusion_count = 0
    duplicate_underlying_case_count = 0
    frozen_exposure_failure_count = 0
    frozen_exposure_attempt_count = 0
    insufficient_transcript_anchor_count = 0
    insufficient_ocr_anchor_count = 0
    quality_exclusions: Counter = Counter()
    source_cases: List[SourceCase] = []
    for row_index, row in iter_jsonl_rows(train_lock.source_path):
        source_row_count += 1
        try:
            source_case = _source_case_from_row(row, row_index)
        except DatasetBuildError:
            malformed_source_row_count += 1
            continue
        source_cases.append(source_case)
    if source_row_count == 0:
        raise DatasetBuildError("authoritative Train source contains no records")
    if not source_cases:
        raise DatasetBuildError(
            "no Train rows match the required Frozen G1 request schema"
        )

    source_cases.sort(
        key=lambda case: (
            case.canonical_underlying_case_id.casefold(),
            case.source_case_id.casefold(),
            case.source_row_index,
        )
    )
    unique_cases: List[SourceCase] = []
    seen_underlying = set()
    for source_case in source_cases:
        key = source_case.canonical_underlying_case_id.casefold()
        if key in seen_underlying:
            duplicate_underlying_case_count += 1
            continue
        seen_underlying.add(key)
        unique_cases.append(source_case)

    eligible: List[EligibleCase] = []
    heldout_excluded_ids = set()
    for source_case in unique_cases:
        canonical = source_case.canonical_underlying_case_id
        if canonical.casefold() in heldout_casefold:
            heldout_exclusion_count += 1
            heldout_excluded_ids.add(canonical)
            continue
        if row_has_ambiguous_provenance(source_case.candidates):
            ambiguous_provenance_exclusion_count += 1
            continue
        frozen_exposure_attempt_count += 1
        try:
            exposed = exposure_adapter.normalize(source_case.request)
        except FrozenExposureUnavailableError:
            raise
        except DatasetBuildError:
            raise
        except Exception:
            frozen_exposure_failure_count += 1
            continue
        candidates = tuple(
            _candidate_from_exposed(record, index)
            for index, record in enumerate(exposed.candidate_units)
        )
        candidate_ids = [candidate.unit_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DatasetBuildError("duplicate model-exposed candidate IDs")
        transcript_anchors, transcript_exclusions = _select_anchor_groups(
            canonical, "TRANSCRIPT", candidates
        )
        ocr_anchors, ocr_exclusions = _select_anchor_groups(canonical, "OCR", candidates)
        quality_exclusions.update(transcript_exclusions)
        quality_exclusions.update(ocr_exclusions)
        if len(transcript_anchors) < 2:
            insufficient_transcript_anchor_count += 1
        if len(ocr_anchors) < 2:
            insufficient_ocr_anchor_count += 1
        if len(transcript_anchors) < 2 or len(ocr_anchors) < 2:
            continue
        eligible.append(
            EligibleCase(
                source_dataset=source_case.source_dataset,
                source_case_id=source_case.source_case_id,
                canonical_underlying_case_id=canonical,
                source_row_index=source_case.source_row_index,
                source_candidate_count=exposed.source_candidate_count,
                truncated_count=exposed.truncated_count,
                dropped_unsupported_count=exposed.dropped_unsupported_count,
                candidates=candidates,
                transcript_anchors=(transcript_anchors[0], transcript_anchors[1]),
                ocr_anchors=(ocr_anchors[0], ocr_anchors[1]),
            )
        )
    if (
        frozen_exposure_attempt_count > 0
        and frozen_exposure_failure_count == frozen_exposure_attempt_count
    ):
        raise FrozenExposureUnavailableError(
            "Frozen exposure normalization failed for every attempted Train case"
        )
    eligible.sort(
        key=lambda case: (
            case.source_dataset.casefold(),
            case.canonical_underlying_case_id.casefold(),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_manifest = {
        "schema_version": SCHEMA_VERSION,
        "freeze_stage": "BEFORE_CALIBRATION_SPLIT_AND_TARGET_GENERATION",
        "eligible_case_count": len(eligible),
        "eligible_cases": [_manifest_case(case) for case in eligible],
    }
    artifact_hashes: Dict[str, str] = {}
    eligible_hash = _write_json(
        output_dir / "eligible_case_manifest.json", eligible_manifest
    )
    artifact_hashes["eligible_case_manifest.json"] = eligible_hash
    _write_sidecar(output_dir / "eligible_case_manifest.sha256", eligible_hash)

    assignments = assign_case_disjoint_splits(eligible)
    if {identity.casefold() for identity in assignments} & heldout_casefold:
        raise DatasetBuildError("held-out identity entered calibration split")
    train_cases = {
        identity for identity, split in assignments.items() if split == "train"
    }
    dev_cases = {identity for identity, split in assignments.items() if split == "dev"}
    if train_cases & dev_cases:
        raise DatasetBuildError("calibration Train/Dev underlying-case overlap")
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_unit": "canonical_underlying_case_id",
        "split_policy": "per-dataset stable SHA256 order; every fifth case to dev",
        "assignments": [
            {
                "canonical_underlying_case_id": case.canonical_underlying_case_id,
                "source_dataset": case.source_dataset,
                "calibration_split": assignments[case.canonical_underlying_case_id],
            }
            for case in eligible
        ],
    }
    split_hash = _write_json(
        output_dir / "calibration_split_manifest.json", split_manifest
    )
    artifact_hashes["calibration_split_manifest.json"] = split_hash
    _write_sidecar(output_dir / "calibration_split_manifest.sha256", split_hash)

    train_examples: List[Dict[str, Any]] = []
    dev_examples: List[Dict[str, Any]] = []
    for case in eligible:
        split = assignments[case.canonical_underlying_case_id]
        target = train_examples if split == "train" else dev_examples
        anchors = (
            (("TRANSCRIPT", anchor) for anchor in case.transcript_anchors),
            (("OCR", anchor) for anchor in case.ocr_anchors),
        )
        for group in anchors:
            for modality, anchor in group:
                target.append(
                    make_calibration_example(
                        case,
                        split,
                        modality,
                        anchor,
                        train_lock.source_sha256,
                        phase4a_config_sha256,
                    )
                )
    for examples in (train_examples, dev_examples):
        examples.sort(key=lambda item: item["calibration_example_id"])
        for example in examples:
            candidate_ids = [item["unit_id"] for item in example["candidate_units"]]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise DatasetBuildError("duplicate candidate IDs in calibration example")
            if not example["positive_unit_ids"]:
                raise DatasetBuildError("calibration example has zero positives")
            assert_no_forbidden_output_fields(example)

    source_inventory = {
        "schema_version": SCHEMA_VERSION,
        "authoritative_train_variant_path": str(train_lock.source_path),
        "authoritative_train_variant_sha256": train_lock.source_sha256,
        "source_row_count": source_row_count,
        "parsed_source_case_count": len(source_cases),
        "unique_underlying_case_count": len(unique_cases),
        "duplicate_underlying_case_count": duplicate_underlying_case_count,
        "malformed_source_row_count": malformed_source_row_count,
        "selection_outputs_inspected": False,
        "veracity_labels_inspected": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
    }
    heldout_payload = {
        "schema_version": SCHEMA_VERSION,
        "step25b_heldout_case_ids": list(selected_heldout),
        "additional_heldout_case_ids": list(additional_heldout),
        "all_heldout_case_ids": sorted(heldout),
        "excluded_source_case_ids_observed": sorted(heldout_excluded_ids),
        "heldout_overlap_after_build": [],
    }
    dataset_case_counts = Counter(case.source_dataset for case in eligible)
    dataset_split_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"train": 0, "dev": 0}
    )
    for case in eligible:
        dataset_split_counts[case.source_dataset][
            assignments[case.canonical_underlying_case_id]
        ] += 1
    all_examples = train_examples + dev_examples
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "authoritative_train_variant_path": str(train_lock.source_path),
        "authoritative_train_variant_sha256": train_lock.source_sha256,
        "expected_train_variant_sha256": expected_train_sha256,
        "phase4a_config_sha256": phase4a_config_sha256,
        "source_row_count": source_row_count,
        "unique_underlying_case_count": len(unique_cases),
        "heldout_case_count": len(heldout),
        "heldout_case_ids": sorted(heldout),
        "heldout_source_case_exclusion_count": heldout_exclusion_count,
        "ambiguous_provenance_exclusion_count": ambiguous_provenance_exclusion_count,
        "malformed_source_row_count": malformed_source_row_count,
        "duplicate_underlying_case_count": duplicate_underlying_case_count,
        "frozen_exposure_failure_count": frozen_exposure_failure_count,
        "frozen_exposure_attempt_count": frozen_exposure_attempt_count,
        "insufficient_transcript_anchor_count": insufficient_transcript_anchor_count,
        "insufficient_ocr_anchor_count": insufficient_ocr_anchor_count,
        "eligible_case_count": len(eligible),
        "calibration_train_case_count": len(train_cases),
        "calibration_dev_case_count": len(dev_cases),
        "calibration_train_example_count": len(train_examples),
        "calibration_dev_example_count": len(dev_examples),
        "ocr_example_count": sum(item["expected_modality"] == "OCR" for item in all_examples),
        "transcript_example_count": sum(
            item["expected_modality"] == "TRANSCRIPT" for item in all_examples
        ),
        "positive_target_count": sum(
            sum(candidate["relevance_target"] for candidate in item["candidate_units"])
            for item in all_examples
        ),
        "negative_target_count": sum(
            sum(1 - candidate["relevance_target"] for candidate in item["candidate_units"])
            for item in all_examples
        ),
        "dataset_case_counts": dict(sorted(dataset_case_counts.items())),
        "dataset_split_counts": {
            dataset: counts
            for dataset, counts in sorted(dataset_split_counts.items())
        },
        "quality_exclusion_reason_counts": dict(sorted(quality_exclusions.items())),
        "selection_outputs_inspected": False,
        "veracity_labels_inspected": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "training_started": False,
        "production_or_model_code_changed": False,
    }

    artifact_hashes["source_inventory.json"] = _write_json(
        output_dir / "source_inventory.json", source_inventory
    )
    artifact_hashes["heldout_exclusions.json"] = _write_json(
        output_dir / "heldout_exclusions.json", heldout_payload
    )
    train_hash = _write_bytes(
        output_dir / "calibration_train.jsonl", _jsonl_bytes(train_examples)
    )
    artifact_hashes["calibration_train.jsonl"] = train_hash
    _write_sidecar(output_dir / "calibration_train.sha256", train_hash)
    dev_hash = _write_bytes(
        output_dir / "calibration_dev.jsonl", _jsonl_bytes(dev_examples)
    )
    artifact_hashes["calibration_dev.jsonl"] = dev_hash
    _write_sidecar(output_dir / "calibration_dev.sha256", dev_hash)
    card = _dataset_card(report).encode("utf-8")
    artifact_hashes["dataset_card.md"] = _write_bytes(output_dir / "dataset_card.md", card)
    report["artifact_sha256"] = artifact_hashes
    _write_json(output_dir / "build_report.json", report)
    return BuildResult(output_dir=output_dir, build_report=report)
