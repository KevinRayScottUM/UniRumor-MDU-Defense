# Step 2.6R-3C0 / 3C1 protocol only

Implementation revision: `step2.6r-3c1-v1`.
Repair: `G1-RelevanceSelector-NaturalPairwise-v1`.

The static, version-controlled `step2_6r_3c1_preregistration.json` records the
user-supplied valid scientific 3B3-R1 FAIL and freezes the next repair protocol.
It does not reinterpret the result as a pipeline failure. The reported result
was copied from the task specification; no real result artifact was opened.
No real dataset builder, review-packet generator, trainer, model loader,
ranking implementation, or deployment switch is included.

From the Defense repository, run the score-free protocol check with:

```sh
.venv/bin/python -m scripts.selector_relevance_next_repair_protocol.run_protocol_preflight
```

It reads only the static preregistration JSON and prints
`NEXT_REPAIR_PROTOCOL_PREFLIGHT_PASS`. A canonical SHA-256 constant in
`protocol.py` freezes the entire JSON content, including types, lists, numerical
thresholds, prior-result provenance, and prohibitions. Duplicate JSON keys and
any change to the frozen content fail closed. The CLI accepts no data, training,
scoring, output, seed, threshold, quarantine-override, or deployment arguments.
The preflight does not create files or load Torch. All dependencies are Python
standard-library modules.

Protocol PASS is only a protocol integrity result. It explicitly reports
`real_exclusion_lock_status=REQUIRED_NOT_SUPPLIED`, `real_cohort_ready=false`,
and `deployment_remains_blocked=true`. The complete real 30-case identity set
has not been supplied here. No real quarantine manifest has been fabricated.
Future identity-only manifests must be supplied and their source artifact hashes
verified against the closed upstream artifacts in a separately authorized stage.
The in-memory validators verify manifest counts, identity digests and structure;
they cannot authenticate an external source file they have not been given.

## Permanent quarantine and exclusion lock

All 30 revealed 3B1/3B2/3B3 audit identities are permanently forbidden for
training, calibration, hyperparameter/seed/epoch/loss/architecture/threshold
selection, model selection, and future acceptance gating. Only descriptive
failure analysis, reporting, reproducibility checks, and integrity verification
are allowed. The exclusion API has no override flag.

`ExclusionLock` requires all four identity-only manifests, with exact unique
identity counts: calibration 1306, revealed audit 30, sealed challenge six,
Stage-A seven. Each locks the identity-set SHA-256 and originating artifact
SHA-256. All supplied `future:<name>` manifests participate. The sequential
order is calibration, revealed audit, sealed challenge, Stage-A, then future
manifests sorted by name. Reports distinguish full locked identity counts,
membership in the supplied inventory, effective sequential exclusions,
pairwise overlaps, and the unique excluded union. Exclusions never silently
consume multiple effective slots for one identity. Formal Validation/Test are
rejected as source partitions and as protocol input paths.

## Frozen future development protocol

Authoritative G1 Train supplies 120 cases, 60 per dataset. The immutable split
contains 96 repair Train cases (48/48) and 24 repair Dev cases (12/12), with
case identities disjoint. The SHA-256 salt is
`step2.6r-3c1-natural-pairwise-v1`. Hash compact UTF-8 JSON
`[salt, purpose, dataset, canonical_case_id]` with `ensure_ascii=false`, sort
by digest then identity, and take the first 60 eligible unexcluded identities
per dataset for purpose `repair-development`. Independently order those 60
using purpose `repair-split`; the first 48 are Train, the remaining 12 Dev.
Insufficient eligible cases block construction; there is no fallback or
resampling. Claims, labels, candidate text, and scores never enter the hash.

Candidate exposure remains authoritative Phase4A normalization, 6–24 units.
Natural claim, unit ID/type/modality/text, and order must match the supplied
immutable source snapshot exactly. Accepted pairs are evidence/text,
title_span/text, transcript/text, and ocr/ocr. Visual pairs are rejected.
`validate_development_cohort` checks already-supplied in-memory manifests; it
does not emit selected data or reviewer packets. Its source inventory is the
eligible Phase4A-exposed Train inventory, not raw unnormalized records.

Independent human A and B review DIRECT/RELATED/IRRELEVANT/UNREADABLE while
blind to scores, veracity labels, dataset identities, other reviewer outputs,
and modality identity where practical. Every four-class disagreement goes to
independent C. Confidence cannot break ties. DIRECT maps to 1; all other labels
map to 0. No model score, veracity label, or lexical anchor generates targets.

Zero-DIRECT cases stay frozen, contribute no pairwise loss, and remain in
reports while excluded from ranking denominators. Overall coverage >=96/120,
Train >=77/96, and Dev >=20/24 must all pass; failure blocks training without
resampling. Although 96 passes the overall condition, both split conditions
together imply at least 97 overall. The validator reports each condition and
their conjunction explicitly.

The encoder and veracity head stay frozen at the registered G1 checkpoint.
Only `selection_head.weight` and `.bias` may train. Sequence A is the original
natural claim; sequence B is exactly
`[UNIT_TYPE=<unit_type>] [MODALITY=<modality>] <candidate text>`.
No neutral or synthetic claim replacement is permitted.

For each case, P is all DIRECT units and N all non-DIRECT units. For every
pair, d = s_positive - s_negative and loss = softplus(-d). Average all pairs
within each case, then average evaluable cases within the batch. There is no
veracity supervision or modality/dataset/source/confidence weighting. The objective is represented as
protocol metadata only; this package does not compute model losses or train.

Future seeds are exactly 42/43/44, at most 10 epochs, AdamW, learning rate
1e-3, weight decay 0, patience 2. Select each seed's epoch by Dev MRR, then
NDCG@5, Recall@5, then earlier epoch. Select the candidate seed by those same
three Dev metrics, then lower seed only as final tie-break. Only the new
24-case Dev split may determine these choices. Development reporting uses
MRR/NDCG@5/Recall@1/3/5, stable candidate-order ties, macro over evaluable
cases, overall and both datasets.

## Later stages remain separate

A new deterministic label-free Train replay cohort must exclude development
Train/Dev, the old audit, historical challenge, Stage-A, calibration, and future
exclusions. It verifies unchanged candidate IDs/order, encoder outputs,
veracity logits, probabilities and predictions; prediction mismatches must be
zero. Selection rank may change. No relevance labels participate. This request
does not specify replay cohort size; that must be frozen before its separately
authorized construction.

Only after development completes, the chosen seed/artifact are frozen, and
invariance passes may a new independent 30-case audit (15/15) be constructed.
Exclude calibration, the old audit, all 120 development cases, historical
challenge, Stage-A, new invariance identities, all other exposed scientific
cases, future exclusions, and Formal Validation/Test. Independent human A/B/C
review and final-gold freeze precede viewing any model scores. Scoring is once.

All future acceptance conditions remain: >=24 evaluable; new overall MRR and
NDCG@5 each strictly better than original G1 on that same audit;
max(delta MRR, delta NDCG@5) >=0.05; delta Recall@5 >=-0.02;
each dataset delta MRR >=-0.05; prediction architecture unchanged. Recall@1/3
are descriptive; there is no CPAC, regression-count, or modality gate and no
gate invention after scoring. Future PASS still requires separate downstream
authorization for production/UI/default/checkpoint changes.
