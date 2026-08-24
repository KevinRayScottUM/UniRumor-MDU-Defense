"""Conservative lexical evidence-sufficiency gate for claim/video inputs.

This module never assigns a scientific verdict.  It only decides whether
multiple evidence channels are clearly unrelated to a sufficiently specific
claim, in which case the production runner can abstain before Frozen G1.
"""

import re
import unicodedata
from enum import Enum
from typing import FrozenSet, Iterable, Sequence

from schemas import RuntimeUnit, SourceType


CLAIM_VIDEO_MISMATCH_WARNING = (
    "Claim-video consistency mismatch; verification abstained."
)


class ConsistencyResult(str, Enum):
    PASS = "pass"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class ClaimConsistencyGate:
    """Apply a conservative, dependency-free lexical consistency check."""

    _TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
    _STOPWORDS = frozenset(
        {
            "a",
            "about",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "been",
            "being",
            "by",
            "claim",
            "claims",
            "clip",
            "do",
            "does",
            "for",
            "frame",
            "from",
            "has",
            "have",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "or",
            "says",
            "scene",
            "show",
            "showing",
            "shown",
            "shows",
            "that",
            "the",
            "their",
            "this",
            "to",
            "video",
            "was",
            "were",
            "with",
        }
    )
    _UNCERTAINTY_MARKERS = frozenset(
        {
            "appears",
            "could",
            "maybe",
            "may",
            "might",
            "perhaps",
            "possibly",
            "seems",
            "someone",
            "something",
            "somewhere",
            "unclear",
            "unknown",
            "unsure",
        }
    )

    @classmethod
    def _tokens(cls, text: str) -> FrozenSet[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return frozenset(cls._TOKEN_PATTERN.findall(normalized))

    @classmethod
    def _informative_tokens(cls, text: str) -> FrozenSet[str]:
        return frozenset(
            token
            for token in cls._tokens(text)
            if token not in cls._STOPWORDS and not token.isdigit()
        )

    @staticmethod
    def _validate_units(
        units: Sequence[RuntimeUnit], expected_source: SourceType
    ) -> None:
        for unit in units:
            if not isinstance(unit, RuntimeUnit):
                raise TypeError("consistency evidence must contain RuntimeUnit values")
            if unit.source_type is not expected_source:
                raise ValueError("consistency evidence source type mismatch")

    @classmethod
    def _source_tokens(cls, units: Iterable[RuntimeUnit]) -> FrozenSet[str]:
        tokens = set()
        for unit in units:
            tokens.update(cls._informative_tokens(unit.text))
        return frozenset(tokens)

    def evaluate(
        self,
        claim: str,
        transcript_units: Sequence[RuntimeUnit],
        ocr_units: Sequence[RuntimeUnit],
        visual_units: Sequence[RuntimeUnit],
    ) -> ConsistencyResult:
        """Return only an evidence-routing state, never a FAKE/REAL verdict."""

        if not isinstance(claim, str):
            raise TypeError("claim must be a string")
        self._validate_units(transcript_units, SourceType.TRANSCRIPT)
        self._validate_units(ocr_units, SourceType.OCR)
        self._validate_units(visual_units, SourceType.VISUAL_OBSERVATION)

        raw_claim_tokens = self._tokens(claim)
        if raw_claim_tokens & self._UNCERTAINTY_MARKERS:
            return ConsistencyResult.UNKNOWN

        claim_tokens = self._informative_tokens(claim)
        source_tokens = (
            self._source_tokens(transcript_units),
            self._source_tokens(ocr_units),
            self._source_tokens(visual_units),
        )
        evidence_tokens = frozenset().union(*source_tokens)

        if claim_tokens & evidence_tokens:
            return ConsistencyResult.PASS

        informative_sources = sum(bool(tokens) for tokens in source_tokens)
        if (
            len(claim_tokens) >= 2
            and len(evidence_tokens) >= 4
            and informative_sources >= 2
        ):
            return ConsistencyResult.MISMATCH
        return ConsistencyResult.UNKNOWN
