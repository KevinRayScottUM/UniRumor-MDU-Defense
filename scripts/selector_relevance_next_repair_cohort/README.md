# Step 2.6R-3C2-A: score-blind repair-development cohort

Implementation: `step2.6r-3c2a-v1`. Repair family:
`G1-RelevanceSelector-NaturalPairwise-v1`.

This package constructs future repair-development data. It does not train,
load a selector checkpoint, instantiate Frozen G1, score a selector, calculate
ranking metrics, or generate relevance annotations. Local tests use synthetic
fixtures only. No real DICC build is part of implementation or regression.
The frozen 3C1 JSON and existing scientific packages remain unchanged.

## Authority and input resolution

`load_preregistration()` from the existing 3C1 protocol package enforces canonical
SHA-256 `81caac242f486eee630cb34c9009482065bcfab35ac920a799f097d9128bceff`.
Sampling targets, candidate limits/pairs, salts, split sizes and exclusion
counts are read from that payload. Its `prior_result` section is never used for
construction. The original protocol validator independently validates the final
membership, split, claim and candidate content/order.

The authoritative Train hash is
`e807535556441434df0ef53a37921c0bdac5e27215ed045104ac08f38275e406`;
composition is 1,636 GroundLie360 plus 2,242 TRUE-3MFact cases. The builder resolves
Train from the 3B1 `cohort_source_lock.json` and the existing `verify_train_lock()`
function. There is no guessed dataset path or CLI hash/fixture override.
All Train rows require an explicit top-level Train split and consistent unique
canonical identities. Formal Validation/Test paths are rejected before reads.
Symlink inputs are rejected so a lock cannot silently change its target.

The old 3B1 directory has an explicit, closed read allowlist:

- `build_report.json` and `build_report.sha256`;
- `cohort_source_lock.json` and `cohort_source_lock.sha256`;
- `selected_case_manifest.json` and `selected_case_manifest.sha256`.

The selected manifest must match the exact tracked 3B1 identity/metadata schema;
claims, labels, scores or additional fields are invalid. It is only an exclusion
input. No old requests, reviewer returns, relevance gold, rankings or gate reports
are opened. Every supplied path is screened before any input is opened, including
`ranking_scores.jsonl`, `per_case_ranking_metrics.jsonl`, `selector_metrics.json`,
`repair_verification_gate_report.json` and `one_shot_evaluation_report.json`.

The closure directory permits only the following four reads:

- `step3b3_scientific_summary.json` and its `.sha256` sidecar;
- `step3b3_closure_manifest.json` and its `.sha256` sidecar.

Their required file hashes are respectively
`6ed3401614f68e58ae0efb3a1671f9f2f7b6b8b653f2100660199ee72938fdf0` and
`71732d7c2b84856ed735cc6d81e7da8af3495fdfba49d9b71cd57df7ed92f81f`.
The summary status must be `STEP_2_6R_3B3_CLOSED_VALID_SCIENTIFIC_FAIL`.
Required boolean flags, wherever present in the two closure documents, must
consistently establish a valid scientific failure, blocked deployment, revealed
audit, and prohibition of training and future acceptance-gate reuse. Closure
metadata authorizes identity quarantine only; metric values never enter any
construction decision. A missing flag or unknown schema fails closed.

## Exclusion lock

The immutable `ExclusionLock` and `ExclusionManifest` classes are reused from 3C1.
The exact order is:

1. `prior_calibration`: 1,306 unique cases, 570 GroundLie360 / 736 TRUE-3MFact;
2. `revealed_audit`: 30 unique cases, 15 / 15;
3. `sealed_challenge`: the existing six identity-only constants;
4. `stage_a_replay`: the existing seven retained identities;
5. all supplied `future:*` manifests, sorted by manifest name.

Calibration identities are projected from the two CLOSED neutral JSONL files.
Their exact frozen hashes are reused by reading the existing trainer's literal
`AUTHORITATIVE_SOURCE_HASHES` constant via AST, without importing the trainer.
The neutral report's status, revision and declared Train/Dev hashes are checked.
The historical neutral writer emitted no report sidecar and 3B1 did not lock that
report: its bytes are therefore newly locked by this preflight. The two JSONLs
retain both their existing sidecars and their 3B1 source-lock bindings. The
neutral revision manifest, pseudo-label values and candidate content are unused.
The actual 3B1 source-lock artifact is the calibration manifest's aggregate
provenance reference, binding both source JSONLs.

Stage-A identities come from the already locked normalization manifest's
`retained_requests` records. The 3B1 source lock binds this manifest and the
supplied invariance report; the report also binds the manifest hash. Historical,
source and canonical identity mappings must agree with the existing seven
constants. The replay JSONL, historical eight-case content and sealed six-case
content are never opened. No Stage-A ranking or prediction values are needed.

Future JSON manifests require exactly:
`name`, `canonical_case_ids`, `identity_sha256`, `source_artifact_sha256`.
`name` must start with `future:`; IDs must be unique canonical identities. The
identity hash uses the frozen 3C1 encoding. A file SHA sidecar is also mandatory.
The supplied file itself is recorded in the construction source lock; the
source-artifact digest is provenance metadata, not a path to follow. Duplicate
names, missing files/sidecars, schema additions and hash mismatches fail closed.

Accounting distinguishes full membership from authoritative-Train intersection,
outside-Train membership, effective first-match counts and pairwise overlaps,
including dataset counts. The existing `unique_excluded_union_count` and
`effective_sequential_count` retain their frozen Train-universe meanings. Explicit
`*_total_count` / `*_outside_train_count` fields report whole-set accounting.
Overlaps are valid set membership; they never cause double-counting or replacement.

## Exposure, privacy and deterministic sampling

The builder uses the existing
`Phase4ANormalizationExposureAdapter.from_project_root()` implementation, which
calls the real external Phase4A `normalize_request`. No simplified production
normalizer exists here. The preflight parses/hashes the four required Phase4A
source files without importing them or running exposure. Build requires exactly
the preflight-approved source bytes. Model/source dependencies are imported by
the existing adapter only for pure normalization; no forward or model loading
entry point is called. Bytecode writes are disabled during that external import.

Source scanning projects identities first. Excluded rows are recorded as
`SKIPPED_EXCLUDED`; their claim/candidate values are not deserialized or exposed.
Hash verification necessarily streams the complete locked Train file's raw bytes.
A syntax-only JSON scanner skips unwanted values without materializing content.
For unexcluded rows, the source veracity label and any unused score/review fields
are projected out. Their values cannot affect sampling or review ordering.

The existing `assess_source_case_provenance()` implements the narrow GroundLie
same-case inherited `:test:` unit-ID exception. It is accepted only with the exact
authoritative Train SHA, GroundLie360 dataset, top-level Train split and exact
same-case historical unit-ID pattern. Non-unit-ID ambiguity remains forbidden.

Eligibility requires successful authoritative exposure, 6–24 candidates,
nonblank exact metadata/text, unique unit IDs and only these pairs:
`evidence/text`, `title_span/text`, `transcript/text`, `ocr/ocr`.
Visual, image and unknown pairs fail closed. Normalization must preserve the
original ordered prefix up to 24 units exactly, with correct truncation accounting
and no unsupported-unit dropping. Deletion, reordering, mutation, added projected
fields or runtime/interface drift is invalid. A plain normalization `ValueError`
for an otherwise valid request is recorded as an exposure failure; integrity and
adapter-contract errors abort construction with exit 2.

Only identity enters selection:

```python
sha256(json.dumps(
    [salt, purpose, dataset, canonical_case_id],
    ensure_ascii=False, separators=(",", ":")
).encode("utf-8")).hexdigest()
```

The frozen salt is `step2.6r-3c1-natural-pairwise-v1`. Within each dataset, sort by
`(digest, canonical_case_id)` using purpose `repair-development` and take the first
60 eligible unexcluded cases. Independently sort those 60 with purpose
`repair-split`: first 48 become `repair_train`, remaining 12 become `repair_dev`.
This yields 120 total, 60/60 by dataset, 96/24 by split, 48/48 Train and 12/12 Dev,
with zero case overlap. Fewer than 60 eligible cases in either dataset freezes a
construction BLOCKED report; no partial cohort, fallback or resampling is allowed.

Private request records contain exactly:
`development_case_id`, `dataset`, `canonical_case_id`, `original_case_id`,
`repair_split`, `claim`, `candidate_units`.
Each candidate contains `unit_id`, `unit_type`, `modality`, `text`,
`original_candidate_position` (zero-based). Natural claims, candidate text and
metadata are retained exactly, including whitespace and Unicode.

The manifest separately locks dataset, canonical/original IDs, split, source row
index, sampling/split hashes, exposed count and ordered unit IDs/types/modalities.
It has no candidate text, labels or scores. The eligibility inventory contains
identity, source row index and accounting only.

## Blank reviewer packets

A and B have the same seven public columns:

```text
review_case_id,claim,review_unit_id,candidate_text,direct_relevance_label,review_confidence,review_note
```

The last three columns are empty. No labels are created. Future annotation classes
remain DIRECT / RELATED / IRRELEVANT / UNREADABLE from frozen 3C1.
Dataset, source IDs, repair split, source row index, unit type/modality, source unit
IDs and scores are absent from the public schema. Original natural-language
content is preserved without redaction or synthetic claim replacement.

Frozen salts are `step2.6r-3c2a-reviewer-a-v1` and
`step2.6r-3c2a-reviewer-b-v1`. Separate hash purposes produce full SHA-256 case IDs,
unit IDs, case ordering and within-case ordering. Public IDs and case order must
differ between A/B. Private mapping rows contain exactly:
`reviewer`, `review_case_id`, `review_unit_id`, `dataset`, `canonical_case_id`,
`original_case_id`, `repair_split`, `unit_id`, `original_candidate_position`.
The mapping envelope records the frozen salts. It contains no annotation or model
output. The mappings reconstruct the same complete candidate set independently.
There is no Reviewer C artifact in this stage.

## CLI modes, artifacts and atomicity

`--preflight` validates source/closure/exclusion locks, Train identities/provenance,
configuration and normalization source availability. It creates no cohort or
reviewer packet and does not import external Phase4A or execute exposure.

`--build-cohort` requires `--approved-preflight-report`. The report, both preflight
locks and their sidecars must exist and exactly match freshly computed canonical
bytes for the current inputs and implementation. An old or modified preflight
cannot approve a changed source, future exclusion set, configuration or builder.
The CLI has no model, score, gold, seed, threshold, resampling, overwrite, expected
hash or synthetic-adapter option. Unit tests patch synthetic trust anchors only
inside the test process; production entry points expose no such overrides.

Preflight produces these three artifacts, each with a same-stem `.sha256` sidecar:

- `cohort_source_preflight_report.json`
- `cohort_source_lock.json`
- `exclusion_source_lock.json`

A successful build produces these nine artifacts, each with a same-stem sidecar:

- `development_cohort_build_report.json`
- `development_cohort_source_lock.json`
- `exclusion_lock.json`
- `eligibility_inventory.json`
- `repair_development_manifest.json`
- `repair_development_requests.jsonl`
- `reviewer_A_template.csv`
- `reviewer_B_template.csv`
- `review_mapping_private.json`

A BLOCKED build produces only the build report, source lock, exclusion lock and
eligibility inventory, plus their four sidecars. It never emits a PASS manifest.
Reports include source composition; all exclusion/overlap accounting; exposure
attempt/failure/skipped counts; below-minimum/valid counts; eligible dataset counts;
selected/split/dataset counts; candidate totals and pair counts; reviewer row counts;
zero label/score counts; and explicit model, training, scoring and forbidden-data
boundary flags. No selector scientific result is produced.

The source lock records `path`, file `sha256` and `role` for every immutable input
actually read: 3C1 JSON, Train source/provenance, config, four external normalization
sources, neutral identity inputs/report, 3B1 identity/provenance artifacts, Stage-A
identity/provenance artifacts, four closure artifacts, future manifests/sidecars,
identity constants and builder/reused implementation files. Build also locks all
three approved preflight artifacts and their sidecars. Referenced unused old
artifacts are not followed. All inputs are rehashed immediately before publication.

Outputs must be under an `outputs/` or `cache/` root, separate from source input
directories. Files are staged in a private sibling temporary directory on the same
filesystem, fsynced, then published with an OS atomic no-replace rename
(Linux `renameat2(RENAME_NOREPLACE)` or macOS `renamex_np(RENAME_EXCL)`). Existing,
partial, symlink or concurrently created destinations are never replaced or
deleted. Only this invocation's temporary directory is cleaned on failure. Other
platforms fail closed. Private directories/files use modes 0700/0600.

Exit codes: **0** valid PASS; **1** atomically frozen feasibility BLOCKED;
**2** invalid input, integrity, runtime or contract failure. A changed scientific
contract cannot be bypassed by CLI flags.

## Proposed DICC commands — do not run locally

The exact old 3B1 cohort subdirectory is not established by repository code.
Discover its identity-only manifest path first. This command lists paths and
performs no cohort construction or artifact-content reads:

```bash
ROOT=/scr/user/kevin2002/TensorCat/uni-rumor
DEFENSE="$ROOT/MDU/Defense_Engineering"
PY=/scr/user/kevin2002/TensorCat/.venv310/bin/python
AUDIT="$DEFENSE/outputs/selector_relevance_independent_audit_v1"
find "$AUDIT" -type f -name selected_case_manifest.json -print
```

After discovery, set `COHORT_DIR` to the exact existing directory printed above
(without the filename). If multiple candidates exist, identify the intended
frozen 3B1 build before invoking preflight; do not guess a dated directory.
The builder itself resolves the exact Train-lock and Stage-A identity-manifest
paths through that existing source lock.

**A. Score-blind preflight**, after setting the discovered `COHORT_DIR`:

```bash
: "${COHORT_DIR:?Set COHORT_DIR to the discovered frozen 3B1 directory}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DEFENSE" "$PY" \
  -m scripts.selector_relevance_next_repair_cohort.run_cohort \
  --preflight \
  --project-root "$ROOT" \
  --phase4a-config "$ROOT/MDU/configs/clip12_phase4a_frozen_g1_inference_handoff.json" \
  --source-3b1-cohort-dir "$COHORT_DIR" \
  --step3b3-closure-dir "$AUDIT/06_one_shot_selector_evaluation_closure" \
  --neutral-dir "$DEFENSE/outputs/selector_relevance_calibration_neutral_v1/01_neutral_revision" \
  --stage-a-invariance-report "$DEFENSE/outputs/selector_relevance_gate_v1/00_prediction_invariance_smoke/prediction_invariance_smoke_report.json" \
  --output-dir "$DEFENSE/outputs/selector_relevance_next_repair_v1/00_cohort_source_preflight"
```

**B. Deterministic build**, only after the resulting preflight is reviewed and
explicitly approved for the later DICC run:

```bash
: "${COHORT_DIR:?Set COHORT_DIR to the discovered frozen 3B1 directory}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DEFENSE" "$PY" \
  -m scripts.selector_relevance_next_repair_cohort.run_cohort \
  --build-cohort \
  --project-root "$ROOT" \
  --phase4a-config "$ROOT/MDU/configs/clip12_phase4a_frozen_g1_inference_handoff.json" \
  --source-3b1-cohort-dir "$COHORT_DIR" \
  --step3b3-closure-dir "$AUDIT/06_one_shot_selector_evaluation_closure" \
  --neutral-dir "$DEFENSE/outputs/selector_relevance_calibration_neutral_v1/01_neutral_revision" \
  --stage-a-invariance-report "$DEFENSE/outputs/selector_relevance_gate_v1/00_prediction_invariance_smoke/prediction_invariance_smoke_report.json" \
  --approved-preflight-report "$DEFENSE/outputs/selector_relevance_next_repair_v1/00_cohort_source_preflight/cohort_source_preflight_report.json" \
  --output-dir "$DEFENSE/outputs/selector_relevance_next_repair_v1/01_repair_development_cohort"
```

Append every explicitly supplied `--future-exclusion-manifest /exact/path.json`
to **both** invocations. No future manifest is guessed. No commands in this
section authorize inference, training, annotation, commit or push.
