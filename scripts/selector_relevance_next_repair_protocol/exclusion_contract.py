"""Identity-only quarantine locks. No scientific artifact reader or override."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence, Tuple

from .schemas import ProtocolError, canonical_identity


REQUIRED_COUNTS = {
    "prior_calibration": 1306,
    "revealed_audit": 30,
    "sealed_challenge": 6,
    "stage_a_replay": 7,
}


def identity_sha256(identities: Sequence[str]) -> str:
    payload = json.dumps(sorted(identities), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExclusionManifest:
    name: str
    canonical_case_ids: Tuple[str, ...]
    identity_sha256: str
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or (self.name not in REQUIRED_COUNTS and not (
            self.name.startswith("future:")
            and self.name[7:].strip()
        )):
            raise ProtocolError("unknown exclusion category")
        if not isinstance(self.canonical_case_ids, tuple) or not self.canonical_case_ids:
            raise ProtocolError("identity-only exclusion manifest must be nonempty and immutable")
        for identity in self.canonical_case_ids:
            canonical_identity(identity)
        if len(self.canonical_case_ids) != len(set(self.canonical_case_ids)):
            raise ProtocolError("duplicate exclusion identities")
        expected = REQUIRED_COUNTS.get(self.name)
        if expected is not None and len(self.canonical_case_ids) != expected:
            raise ProtocolError(f"{self.name} exclusion lock requires exactly {expected} identities")
        if self.identity_sha256 != identity_sha256(self.canonical_case_ids):
            raise ProtocolError("exclusion identity SHA-256 mismatch")
        if not isinstance(self.source_artifact_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_artifact_sha256
        ):
            raise ProtocolError("exclusion source artifact SHA-256 is required")


@dataclass(frozen=True)
class ExclusionLock:
    manifests: Tuple[ExclusionManifest, ...]
    expected_future_manifest_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.manifests, tuple) or any(
            not isinstance(item, ExclusionManifest) for item in self.manifests
        ):
            raise ProtocolError("immutable exclusion manifests are required")
        names = [item.name for item in self.manifests]
        if len(names) != len(set(names)) or not set(REQUIRED_COUNTS) <= set(names):
            raise ProtocolError("complete unique calibration/audit/challenge/Stage-A locks required")
        expected = self.expected_future_manifest_names
        if not isinstance(expected, tuple) or any(
            not isinstance(name, str) or not name.startswith("future:") for name in expected
        ) or not set(expected) <= set(names):
            raise ProtocolError("an explicitly supplied future exclusion manifest is missing")

    @property
    def ordered(self) -> Tuple[ExclusionManifest, ...]:
        by_name = {item.name: item for item in self.manifests}
        future = sorted(set(by_name) - set(REQUIRED_COUNTS))
        return tuple(by_name[name] for name in (*REQUIRED_COUNTS, *future))

    @property
    def all_identities(self) -> frozenset[str]:
        return frozenset(identity for item in self.manifests for identity in item.canonical_case_ids)

    def reject_overlap(self, identities: Sequence[str]) -> None:
        if set(identities) & self.all_identities:
            raise ProtocolError("candidate overlaps mandatory identity exclusion/quarantine lock")

    def accounting(self, universe: Sequence[str]) -> Mapping[str, object]:
        if len(universe) != len(set(universe)):
            raise ProtocolError("exclusion accounting universe contains duplicate identities")
        pool = set(universe)
        seen: set[str] = set()
        groups = []
        memberships = {}
        for item in self.ordered:
            membership = pool.intersection(item.canonical_case_ids)
            memberships[item.name] = membership
            groups.append({
                "name": item.name,
                "locked_identity_count": len(item.canonical_case_ids),
                "identity_sha256": item.identity_sha256,
                "source_artifact_sha256": item.source_artifact_sha256,
                "membership_count": len(membership),
                "effective_sequential_count": len(membership - seen),
            })
            seen.update(membership)
        overlaps = [
            {"first": first, "second": second,
             "overlap_count": len(memberships[first] & memberships[second])}
            for first, second in combinations(memberships, 2)
        ]
        return {"universe_count": len(pool), "categories": groups,
                "pairwise_overlap_counts": overlaps,
                "unique_excluded_union_count": len(seen),
                "remaining_count": len(pool - seen)}
