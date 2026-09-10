# Step 2.6R-3C2-A: score-blind repair-development cohort

Implementation: `step2.6r-3c2a-r3-v1`. Repair family:
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
Generic symlink inputs remain rejected by the unchanged `safe_path()` contract.

R2 accepts one historical provenance alias, only inside the hash-locked Phase3A
Train-lock report's `source.path`. The independent 3B1 `authoritative_g1_train`
record is validated first: its path must be canonical, inside the project root,
and its declared and actual SHA must match the frozen authoritative Train SHA.
All inventory, exposure and build reads use that canonical path of record.

The historical report path may already equal the canonical path. Otherwise it
must have exactly this project-local relationship, with an identical nonempty
relative tail and no parent traversal or additional symlink components:

```text
<project-root>/MDU/outputs/<relative-tail>
  -> <project-root>/MDU/Academic_Research/outputs/<relative-tail>
```

`MDU/outputs` must itself be a symlink directly targeting
`Academic_Research/outputs` (the diagnosed DICC target string) or the exact
absolute canonical directory. Other spellings, intermediate links, different
tails, missing sources, outside-root sources and Formal Validation/Test paths
fail closed. Strict resolution must equal the frozen canonical file path;
matching bytes or inode alone is insufficient. The existing `verify_train_lock()`
is reused unchanged. Its returned source must resolve to the same canonical path
and its returned SHA must match; the canonical file is then hash-checked again.
Neither historical report nor 3B1 source lock nor symlink is rewritten.

Both preflight and development source locks contain
`authoritative_train_provenance`, recording `historical_source_path`,
`historical_source_resolved_path`, `canonical_authoritative_train_path`,
`historical_alias_used`, `historical_alias_component`, and
`historical_alias_link_target`. The component/target are null when no alias is
used. The preflight report binds this metadata through `cohort_source_lock_sha256`;
the build report binds its source lock through `artifact_sha256`.

The source layer's `TrainSourceLedger` checks the exact relationship before and
after normal input hash revalidation. `cohort_builder.py` only changes its ledger
import so the existing preparation and final `freeze()` checks cover this
metadata; `artifacts.py`, `safe_path()` and R1 publication behavior are unchanged.
Build preparation recomputes provenance and requires byte equality with the
approved preflight. Retargeting to another source fails; changing the raw link
target string even while retaining the same resolution invalidates approval.
Changes detected before publication leave no final output. There is no link
repair, generic symlink permission, or override flag. All top-level paths,
other source-lock records, reviewer artifacts, future exclusion manifests and
output paths retain strict path validation.

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

R3 separates raw source structure from final exposed-candidate eligibility. The
R2 preflight applied the final `allowed_pairs` to every raw unit, failing before
exposure with `unsupported/visual source candidate pair`. The supplied DICC
diagnosis found that the strict authoritative normalizer preserves
`review_certified_visual_unit/ocr` in some final pools. That observation is not
an identity exclusion or a special unit-type rule; no diagnostic counts enter
the implementation.

Preflight still checks locked Train identity/provenance, natural claim, list/dict
structure and nonblank required candidate strings. It does not apply the final
pair allowlist to raw candidates and does not import/execute Phase4A. The complete
raw request, with original unit IDs, types, modalities, text and order, is passed
to the unchanged authoritative adapter during build. The existing strict visual
policy invocation remains unchanged; no dropping mode is introduced.

Normalization must preserve the original ordered prefix up to the frozen maximum
24 exactly, with correct source/truncation counts and zero dropped unsupported
units. Deletion, reordering, mutation, added projected fields, duplicate exposed
IDs or runtime/interface drift remain global integrity failures (exit 2). A plain
normalization `ValueError` for an otherwise valid request remains an exposure
failure. These checks finish before candidate eligibility is assessed.

Only the FINAL exposed pool is checked against the frozen pairs:
`evidence/text`, `title_span/text`, `transcript/text`, `ocr/ocr`. If any final pair
is outside that set, the entire case has `eligible=false`, `excluded=false`,
`exclusion_reason=null`, status `INELIGIBLE_EXPOSED_CANDIDATE_CONTRACT` and
`ineligibility_reason=EXPOSED_PAIR_OUTSIDE_FROZEN_ALLOWED_PAIRS`. No unit is dropped,
renamed or coerced, and the full exposed count is retained in the private
inventory. This is neither an identity exclusion nor an exposure failure.
Such cases never enter selection, requests, manifests, A/B packets or mappings.

Only pair-valid cases are then checked against the frozen minimum 6; smaller
pools have `ineligibility_reason=EXPOSED_CANDIDATE_COUNT_BELOW_MINIMUM`. A pool
with five allowed units and one unsupported unit is a six-unit contract-ineligible
case, not a five-unit case. An unsupported raw unit beyond authoritative truncation
does not invalidate a final pool that contains only allowed pairs. No custom
reordering or truncation policy is permitted.

Both eligibility inventory and build report carry score-free/text-free counts:

- `phase4a_exposure_attempt_count`, `phase4a_exposure_failure_count`;
- `exposed_candidate_contract_ineligible_case_count` and
  `exposed_candidate_contract_ineligible_dataset_counts`;
- `exposed_unsupported_unit_count` and `exposed_unsupported_pair_counts`
  (sorted records with `unit_type`, `modality`, `count`);
- `candidate_count_below_6_count`, `candidate_count_valid_count` and
  `eligible_unexcluded_counts`.

Attempts partition into exposure failures, candidate-contract ineligible cases,
pair-valid cases below minimum and eligible cases. Identity exclusions remain
separate skipped cases. Preflight does not determine final eligibility. Existing
build report/inventory assembly carries the new counts without changing the
builder or sampling functions. Fewer than 60 eligible cases in either dataset
uses the existing feasibility BLOCKED publication with no fallback or resampling.

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
filesystem and fsynced before publication. All immutable inputs are still
revalidated by `freeze()` immediately before calling the publication layer.
Private directories/files use modes 0700/0600.

R1 fixes a filesystem portability failure reported on DICC `/scr`: libc exposes
`renameat2`, but the filesystem returns `EINVAL` for `RENAME_NOREPLACE`. macOS
used a different primitive and therefore did not exercise that failure. This
is a pre-build engineering failure. Scientific rules, exclusions, reviewer salts,
candidate exposure and the frozen 3C1 preregistration are unchanged.

Publication prefers Linux `renameat2(RENAME_NOREPLACE)` or macOS
`renamex_np(RENAME_EXCL)`. Only **EINVAL, ENOSYS, EOPNOTSUPP, ENOTSUP** permit
fallback (aliases are deduplicated by the platform's errno values). A missing
native symbol is treated as ENOSYS. EACCES, EPERM, EROFS, ENOSPC, EIO, EXDEV and
other errors remain failures; EEXIST/ENOTEMPTY continue to reject existing output.

Both the native path and fallback acquire the same deterministic sibling lock,
`.<destination-name>.publish.lock`, with
`os.open(O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0o600)` where O_CLOEXEC exists.
Using the lock on both paths prevents a native publisher from bypassing an
active fallback publisher or a stale lock. Working native primitives remain
the preferred rename operation; successful native calls never invoke fallback.

An existing lock causes immediate failure: no waiting, stealing, overwrite flag,
automatic recovery or deletion. A crash can leave a stale lock. Operator
inspection is required before a later retry; the program does not remove it.
The lock descriptor remains open through publication and cleanup. On a normal
handled return/failure, cleanup unlinks only the lock with the same device/inode
as that descriptor. A foreign replacement is left untouched.

While holding the lock, fallback rechecks that the destination does not exist
and is not a symlink, including a dangling symlink. Existing empty/non-empty
directories and partial outputs are preserved. Staging must be a real directory
on the same device as the opened destination parent; otherwise publication fails
with EXDEV. Normal same-filesystem `os.rename` then publishes the complete staged
directory atomically to official readers. There is no copy/move fallback and no
unlocked exists-then-rename sequence. Cooperating publishers must use this package's
lock protocol; arbitrary external writers that ignore the lock are outside that
cooperative guarantee.

The destination parent is fsynced before releasing the lock. Platforms/filesystems
that reject directory fsync with the same explicit unsupported-capability errno
set are tolerated; other sync errors propagate. If a real sync error happens
after rename, the complete published directory is preserved and the error is
reported; the program never deletes it to attempt a rollback. Rename failures
leave no partial final output. `freeze()` only cleans this invocation's own
temporary staging directory on failure. Existing output is never reused or deleted.

Exit codes: **0** valid PASS; **1** atomically frozen feasibility BLOCKED;
**2** invalid input, integrity, runtime or contract failure. A changed scientific
contract cannot be bypassed by CLI flags.

## Proposed R3 DICC commands — report only

Use Bash and the existing DICC environment after a later reviewed commit/push.
No DICC command is executed by this implementation task. Retain the canonical
input variables from the read-only diagnosis; no historical alias is substituted
for a top-level input. Real construction still requires review of the repaired
preflight; no real build command is provided in R3.

```bash
ROOT=/scr/user/kevin2002/TensorCat/uni-rumor
DEFENSE="$ROOT/MDU/Defense_Engineering"
PY=/scr/user/kevin2002/TensorCat/.venv310/bin/python
```

**A. Focused synthetic regression**, including all R1/R2 tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DEFENSE" "$PY" \
  -m unittest discover -s "$DEFENSE/tests" \
  -p 'test_selector_relevance_next_repair_cohort.py' -k ExposureEligibilityBoundaryTests -v &&
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DEFENSE" "$PY" \
  -m unittest discover -s "$DEFENSE/tests" \
  -p 'test_selector_relevance_next_repair_cohort.py' -v
```

**B. Full synthetic regression**:

```bash
(
set -e
cd "$DEFENSE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DEFENSE"
"$PY" -m unittest discover -s tests -p 'test_selector_relevance_next_repair_protocol.py' -v
"$PY" -m unittest discover -s tests -p 'test_selector*.py' -v
"$PY" -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX="$DEFENSE/cache/step2_6r_3c2a_r3_pycache" "$PY" -m compileall -q app schemas services adapters webapp scripts tests
"$PY" -m app.mock_demo
git diff --check
git status --short
)
```

Before C and D, define the Bash array `FUTURE_EXCLUSION_MANIFESTS` containing every
previously supplied canonical future-manifest path. Use
`FUTURE_EXCLUSION_MANIFESTS=()` only if none were supplied. Use the same array and
canonical input variables for both commands. Missing variables fail closed.

**C. Real score-blind source preflight**, only after A/B pass:

```bash
(
set -e
declare -p FUTURE_EXCLUSION_MANIFESTS >/dev/null
future_args=()
for manifest in "${FUTURE_EXCLUSION_MANIFESTS[@]}"; do
  future_args+=(--future-exclusion-manifest "$manifest")
done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DEFENSE" "$PY" \
  -m scripts.selector_relevance_next_repair_cohort.run_cohort \
  --preflight \
  --project-root "${ROOT:?Reuse the verified canonical project root}" \
  --phase4a-config "${PHASE4A_CONFIG:?Reuse the verified canonical Phase4A config}" \
  --source-3b1-cohort-dir "${COHORT_DIR:?Reuse the verified canonical frozen 3B1 directory}" \
  --step3b3-closure-dir "${CLOSURE_DIR:?Reuse the verified canonical closure directory}" \
  --neutral-dir "${NEUTRAL_DIR:?Reuse the verified canonical neutral directory}" \
  --stage-a-invariance-report "${STAGE_A_REPORT:?Reuse the verified canonical Stage-A report}" \
  --output-dir "${PREFLIGHT_DIR:?Reuse the verified canonical preflight output path}" \
  "${future_args[@]}"
)
```

**D. Strict preflight artifact integrity verification**, read-only. Require
exactly three JSON artifacts and their sidecars, strict paths, current source and
implementation hashes, fresh exclusion locks, R2 alias provenance and byte-exact
report/lock agreement. This reuses preflight preparation and input revalidation;
it performs no Phase4A import/exposure, publication or cohort construction.

```bash
(
set -e
declare -p FUTURE_EXCLUSION_MANIFESTS >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DEFENSE" "$PY" \
  - "${ROOT:?}" "${PHASE4A_CONFIG:?}" "${COHORT_DIR:?}" "${CLOSURE_DIR:?}" \
  "${NEUTRAL_DIR:?}" "${STAGE_A_REPORT:?}" "${PREFLIGHT_DIR:?}" \
  "${FUTURE_EXCLUSION_MANIFESTS[@]}" <<'PY'
import json
import sys
from pathlib import Path
from scripts.selector_relevance_next_repair_cohort import artifacts as ar, cohort_builder as cb
from scripts.selector_relevance_next_repair_cohort.schemas import CohortError, Inputs

paths = [Path(value) for value in sys.argv[1:]]
inputs = Inputs(*paths[:6], future_exclusion_manifests=tuple(paths[7:]))
output = ar.safe_path(paths[6])
names = {"cohort_source_preflight_report.json", "cohort_source_lock.json", "exclusion_source_lock.json"}
expected = names | {Path(name).with_suffix(".sha256").name for name in names}
if {path.name for path in output.iterdir()} != expected:
    raise CohortError("preflight artifact set differs from exact six-file contract")
for name in expected:
    if not ar.safe_path(output / name).is_file():
        raise CohortError("preflight artifact is not a canonical regular file")
_, _, _, ledger, source_lock, exclusions, report = cb.prepare(inputs)
cb.approve(output / "cohort_source_preflight_report.json", ledger, source_lock, exclusions, report)
ledger.revalidate()
print(json.dumps({"status": "R3_PREFLIGHT_ARTIFACT_INTEGRITY_PASS",
                  "implementation_revision": report["implementation_revision"],
                  "phase4a_exposure_performed": False, "real_cohort_written": False}))
PY
)
```

The verification command checks engineering provenance only. It does not certify
cohort feasibility or produce a scientific result. Review the repaired preflight
before separately authorizing real construction. No inference, training,
annotation, commit or push is performed by these proposed preflight commands.
