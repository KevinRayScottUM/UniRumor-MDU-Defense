# Step 2.6R-3B1 Independent Relevance Audit

This package builds and freezes an **independent score-blind Train-derived
direct-relevance audit cohort**. It performs label-free deterministic Phase4A
request normalization only. It does not load either selector, Frozen G1, a
checkpoint, Torch, an optimizer, or any prediction output.

Step 2.6R-3B1-R1 repairs authoritative historical Stage-A replay identity
integration only. Stage-A canonical exclusion identities come from the already
verified R3A0-R1 manifest mapping. Historical `smoke::...` replay IDs are not
reparsed with the generic cross-case canonicalizer. No sampling, review, or
scientific evaluation protocol changed.

Step 2.6R-3B1-R2 separates exclusion-set membership from effective first-match
accounting. Exclusion categories may validly overlap: the final exclusion is
their set union, while operational effective counts follow the frozen
`sealed -> Stage A -> calibration -> additional` precedence. The real DICC
identity-only audit found two calibration/Stage-A overlaps; they are valid and
do not indicate leakage. Legacy `*_exclusion_count` fields continue to report
membership-set sizes, and new `*_effective_count` fields report first-match
counts. No exclusion, sampling, reviewer-blinding, or scientific protocol
changed.

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
