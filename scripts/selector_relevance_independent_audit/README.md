# Step 2.6R-3B1 Independent Relevance Audit

This package builds and freezes an **independent score-blind Train-derived
direct-relevance audit cohort**. It performs label-free deterministic Phase4A
request normalization only. It does not load either selector, Frozen G1, a
checkpoint, Torch, an optimizer, or any prediction output.

The original six-case historical challenge remains sealed because explicit
access authorization was not established. Only those six canonical identities
are used as exclusion keys; their claims, candidate pools, annotations, and
rankings are not opened or emitted. A new cohort avoids retroactive heldout
reuse.

Cases are derived from the immutable Frozen G1 Train source, but are disjoint
from selector-calibration Train/Dev cases, the seven Stage-A replay cases, the
sealed six-case challenge, and any explicit identity-only prior-audit manifest.
Construction is selector-score-blind and veracity-label-blind. Public packets
are independently ordered and dataset-blind, modality-blind, selector-blind,
and veracity-label-blind.

This is candidate-rich engineering repair verification, not a representative
population sample and not Formal Validation/Test generalization.

## Frozen build rules

- Verify the authoritative Train SHA and exact 3,878-case composition.
- Reuse the real Phase4A normalization/candidate-exposure contract; no fallback.
- Retain only 6--24 unique exposed text/OCR candidates and reject visual units.
- Rank cases independently within each dataset by
  `SHA256(step2.6r-3b1-independent-audit-v1|dataset|canonical_case_id)`.
- Freeze the first 15 GroundLie360 and first 15 TRUE-3MFact cases.
- Never replace a selected case after inspection or review.

Reviewer templates use exactly `DIRECT`, `RELATED`, `IRRELEVANT`, and
`UNREADABLE`; only `DIRECT` is the future positive relevance target. Reviewer
confidence is descriptive (`HIGH`, `MEDIUM`, or `LOW`) and never changes that
target.

The build stops after packet generation. Step 2.6R-3B2 remains locked until two
completed reviews and provenance hashes are independently frozen. Selector
scoring is not authorized in Step 2.6R-3B1.
