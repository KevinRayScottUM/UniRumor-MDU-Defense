"""Immutable in-memory contracts for synthetic protocol checks only."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


IMPLEMENTATION_REVISION = "step2.6r-3c1-v1"
DATASETS = ("GroundLie360", "TRUE-3MFact")
PAIRS = frozenset({("evidence", "text"), ("title_span", "text"),
                   ("transcript", "text"), ("ocr", "ocr")})
LABELS = frozenset({"DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE"})


class ProtocolError(ValueError):
    """An immutable protocol or supplied identity contract was violated."""


def nonblank(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a nonblank string")


def canonical_identity(value: str) -> None:
    nonblank(value, "canonical_case_id")
    if not re.fullmatch(r"(?:GroundLie360|TRUE-3MFact):[A-Za-z0-9_.-]+", value):
        raise ProtocolError("canonical identity must be dataset:case with no split/path fields")


@dataclass(frozen=True)
class Candidate:
    unit_id: str
    unit_type: str
    modality: str
    text: str

    def __post_init__(self) -> None:
        for field in ("unit_id", "unit_type", "modality", "text"):
            nonblank(getattr(self, field), field)
        if (self.unit_type, self.modality) not in PAIRS:
            raise ProtocolError("unsupported or visual candidate pair")


@dataclass(frozen=True)
class FrozenTrainCase:
    canonical_case_id: str
    dataset: str
    original_claim: str
    candidate_units: Tuple[Candidate, ...]
    source_partition: str = "Train"
    exposure_contract: str = "authoritative Phase4A normalization"

    def __post_init__(self) -> None:
        canonical_identity(self.canonical_case_id)
        if self.dataset not in DATASETS or not self.canonical_case_id.startswith(self.dataset + ":"):
            raise ProtocolError("dataset and canonical identity disagree")
        if self.source_partition != "Train":
            raise ProtocolError("only authoritative Train is permitted; Formal Validation/Test forbidden")
        if self.exposure_contract != "authoritative Phase4A normalization":
            raise ProtocolError("authoritative Phase4A exposure is required")
        nonblank(self.original_claim, "original natural claim")
        if not isinstance(self.candidate_units, tuple) or not 6 <= len(self.candidate_units) <= 24:
            raise ProtocolError("candidate count must be 6-24 in an immutable tuple")
        if any(not isinstance(unit, Candidate) for unit in self.candidate_units):
            raise ProtocolError("candidate contract is required")
        ids = tuple(unit.unit_id for unit in self.candidate_units)
        if len(ids) != len(set(ids)):
            raise ProtocolError("duplicate candidate unit ID")


@dataclass(frozen=True)
class DevelopmentCase:
    case: FrozenTrainCase
    repair_split: str

    def __post_init__(self) -> None:
        if not isinstance(self.case, FrozenTrainCase):
            raise ProtocolError("frozen Train case is required")
        if self.repair_split not in {"repair_train", "repair_dev"}:
            raise ProtocolError("repair split must be repair_train or repair_dev")
