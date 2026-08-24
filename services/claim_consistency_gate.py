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
    _YEAR_PATTERN = re.compile(r"(?<!\d)(?:1\d{3}|2\d{3})(?!\d)")
    _ACRONYM_PATTERN = re.compile(r"(?<!\w)[A-Z][A-Z0-9]{1,9}(?!\w)")
    _PROPER_NAME_PATTERN = re.compile(
        r"(?<!\w)[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
        r"(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){1,3}(?!\w)"
    )
    _GENERIC_ENTITY_TOKENS = frozenset(
        {
            "a",
            "an",
            "claim",
            "clip",
            "frame",
            "scene",
            "the",
            "this",
            "video",
        }
    )
    _ENTITY_TITLES = frozenset(
        {
            "dr",
            "former",
            "governor",
            "minister",
            "mr",
            "mrs",
            "ms",
            "president",
            "professor",
            "senator",
        }
    )
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

    @classmethod
    def _explicit_years(cls, text: str) -> FrozenSet[str]:
        normalized = unicodedata.normalize("NFKC", text)
        return frozenset(cls._YEAR_PATTERN.findall(normalized))

    @classmethod
    def _named_entities(cls, text: str) -> FrozenSet[str]:
        """Extract only explicit acronym or multi-token proper-name mentions."""

        normalized = unicodedata.normalize("NFKC", text)
        entities = set()
        for pattern in (cls._ACRONYM_PATTERN, cls._PROPER_NAME_PATTERN):
            for match in pattern.finditer(normalized):
                entity = " ".join(match.group(0).casefold().split())
                entity_tokens = [
                    token
                    for token in entity.split()
                    if token not in cls._ENTITY_TITLES
                ]
                if entity_tokens and not all(
                    token in cls._GENERIC_ENTITY_TOKENS
                    or token in cls._STOPWORDS
                    for token in entity_tokens
                ):
                    entities.add(" ".join(entity_tokens))
        return frozenset(entities)

    @staticmethod
    def _entities_supported(
        claim_entities: FrozenSet[str], evidence_entities: FrozenSet[str]
    ) -> bool:
        for claim_entity in claim_entities:
            claim_parts = frozenset(claim_entity.split())
            supported = any(
                claim_entity == evidence_entity
                or (
                    len(claim_parts) > 1
                    and claim_parts.issubset(evidence_entity.split())
                )
                for evidence_entity in evidence_entities
            )
            if not supported:
                return False
        return True

    @classmethod
    def _source_attributes(
        cls, units: Iterable[RuntimeUnit]
    ) -> tuple[FrozenSet[str], FrozenSet[str]]:
        years = set()
        entities = set()
        for unit in units:
            years.update(cls._explicit_years(unit.text))
            entities.update(cls._named_entities(unit.text))
        return frozenset(years), frozenset(entities)

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
        evidence_units = tuple(transcript_units) + tuple(ocr_units) + tuple(
            visual_units
        )
        claim_years = self._explicit_years(claim)
        claim_entities = self._named_entities(claim)
        evidence_years, evidence_entities = self._source_attributes(evidence_units)

        if (
            claim_years
            and evidence_years
            and claim_years.isdisjoint(evidence_years)
        ):
            return ConsistencyResult.MISMATCH

        if claim_entities and not self._entities_supported(
            claim_entities, evidence_entities
        ):
            return ConsistencyResult.MISMATCH

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
