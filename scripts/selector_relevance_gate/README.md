# Step 2.6R-3 Held-out Relevance and Prediction-Invariance Gate

This package performs read-only evaluation. It never trains, constructs an
optimizer, computes gradients, reads fake/real labels, or changes production.
The original and calibrated evaluation states share the exact authoritative
Frozen G1 encoder, veracity head, tokenizer, collator, candidate order, and
class-wise maximum sample pooling. Only `selection_head.weight` and
`selection_head.bias` may differ.

Execution is deliberately split and never chained automatically:

1. `--prepare-invariance-requests` performs a standard-library-only field
   projection of the immutable historical Phase4A request artifact. It removes
   the provenance-only `source_case_id` and
   `ground_truth_label_deliberately_omitted` fields, projects each exact rich
   17-field candidate to the four Frozen-G1 input fields, and deterministically
   excludes the later-protected CPAC held-out case. It does not load Torch, a
   model, a checkpoint, or a selector.
2. `--invariance-smoke` evaluates the deterministic seven-request nonheldout
   subset of the immutable historical eight-request Phase4A label-free smoke
   artifact and writes a prediction-invariance report.
3. `--heldout-gate` requires a manually inspected Stage-A PASS report before
   opening the held-out relevance reference artifact.

One historical Phase4A smoke request later became the preregistered CPAC
held-out relevance challenge. It is deterministically excluded from Stage A
before any calibrated Stage-A evaluation so that the Stage-B challenge ranking
remains unrevealed. This result-blind protocol correction was made before any
real Step 2.6R-3 evaluation.

Neither stage may access Formal Validation or Formal Test. Stage B is only
repair-verification on six pre-existing held-out relevance challenge cases; it
is not an untouched final test or a population-level generalization benchmark.

## Required authoritative input schemas

The immutable historical source is:

`MDU/outputs/clip12_phase4a_frozen_g1_end_to_end_mdu_inference_handoff/01_smoke_requests/clip12p4a_smoke_requests.jsonl`

Its required SHA-256 is
`356ee750c7b95de37e5d14b481e2f5f8fb5ae1e3805ee922d016fcb0a3ab2178`.
It contains exactly eight label-free rows with
`{case_id, dataset, claim, candidate_units, source_case_id,
ground_truth_label_deliberately_omitted}`. The omission sentinel must be the
boolean `true` in all eight rows; it records that labels were omitted and is not
a label. The source contains exactly 73 candidates with the exact rich fields
`{claim_atom, evidence_refs, evidence_text, frame_ids, grounding, modality,
phase1_source, relation, snippet_id, snippet_path, snippet_text, snippet_type,
source_snippet_type, supervision, text, unit_id, unit_type}`. Their exact
unit-type/modality accounting is 28 `ocr|ocr`, 10 `title_span|text`, and 35
`transcript|text`.

Normalization projects the seven retained rows to exactly
`{case_id, dataset, claim, candidate_units}` without changing any string,
candidate ID, candidate order, unit type, or modality. Each normalized candidate
contains only `{unit_id, unit_type, modality, text}`; rich upstream metadata is
never passed to Frozen G1. It writes:

- `phase4a_invariance_requests.jsonl`
- `phase4a_invariance_requests.sha256`
- `phase4a_invariance_request_manifest.json`
- `phase4a_invariance_request_manifest.sha256`

The Stage-A loader requires independent SHA-256 values for both the normalized
JSONL and its manifest. The manifest must prove the authoritative historical
SHA, exact source schemas, all-true omission sentinels, 73-unit accounting,
exact unit-type/modality counts, source count 8, exactly one CPAC exclusion,
retained count 7, and zero scientific-content changes. An arbitrary
seven-request artifact is rejected.

The normalized JSONL schema is:

```json
{"case_id":"...","dataset":"...","claim":"...","candidate_units":[{"unit_id":"...","unit_type":"title_span","modality":"text","text":"..."}]}
```

It must contain exactly seven manifest-approved requests and no labels or
stored model outputs.

The held-out reference JSON is also supplied with an independent SHA-256. Its
`artifact_type` is
`preexisting_heldout_relevance_challenge_references`. Each reference contains
the original claim, exact ordered candidates, audited positive IDs, prior
original-selector rank/Top-5 provenance, and the path plus SHA-256 of its
immutable source audit artifact. It must cover exactly:

- `GroundLie360:13025004`
- `TRUE-3MFact:10145403`
- `TRUE-3MFact:10258205`
- `TRUE-3MFact:10372904`
- `TRUE-3MFact:10455808`
- `TRUE-3MFact:10865013`

The evaluator does not generate this artifact or recreate annotations. The
exact existing Stage-B source paths and hashes must still be discovered and
reviewed before the real held-out stage is run.

Useful read-only discovery commands:

```bash
find /scr/user/kevin2002/TensorCat/uni-rumor/MDU -type f \
  \( -iname '*phase4a*replay*.json' -o -iname '*phase4a*request*.jsonl' \) -print

find /scr/user/kevin2002/TensorCat/uni-rumor/MDU/Defense_Engineering/outputs \
  -type f \( -iname '*probe*manifest*.json' -o -iname '*relevance*reference*.json' \
  -o -iname '*reproduction*.json' \) -print
```

Full training remains outside this package and is never triggered by any mode.
