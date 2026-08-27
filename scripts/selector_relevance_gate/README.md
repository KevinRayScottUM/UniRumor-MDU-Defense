# Step 2.6R-3 Held-out Relevance and Prediction-Invariance Gate

This package performs read-only evaluation. It never trains, constructs an
optimizer, computes gradients, reads fake/real labels, or changes production.
The original and calibrated evaluation states share the exact authoritative
Frozen G1 encoder, veracity head, tokenizer, collator, candidate order, and
class-wise maximum sample pooling. Only `selection_head.weight` and
`selection_head.bias` may differ.

Execution is deliberately split:

1. `--invariance-smoke` evaluates exactly eight pre-existing, label-free
   Phase4A replay requests and writes a prediction-invariance report.
2. `--heldout-gate` requires a manually inspected Stage-A PASS report before
   opening the held-out relevance reference artifact.

Neither stage may access Formal Validation or Formal Test. Stage B is only
repair-verification on six pre-existing held-out relevance challenge cases; it
is not an untouched final test or a population-level generalization benchmark.

## Required authoritative input schemas

The Phase4A replay artifact is supplied with an independent SHA-256. The
preferred input is the authoritative native JSONL containing exactly eight
records with `{case_id, dataset, claim, candidate_units}`. The evaluator also
accepts the following explicit JSON wrapper for controlled contract fixtures:

```json
{
  "schema_version": 1,
  "artifact_type": "phase4a_label_free_replay_requests",
  "requests": [
    {
      "request_id": "...",
      "case_id": "...",
      "dataset": "...",
      "claim": "...",
      "candidate_units": [
        {"unit_id": "...", "unit_type": "text", "modality": "text", "text": "..."}
      ]
    }
  ]
}
```

It must contain exactly eight authoritative requests and no labels or stored
model outputs.

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
exact existing DICC source paths and hashes must be discovered and reviewed
before either real stage is run.

Useful read-only discovery commands:

```bash
find /scr/user/kevin2002/TensorCat/uni-rumor/MDU -type f \
  \( -iname '*phase4a*replay*.json' -o -iname '*phase4a*request*.jsonl' \) -print

find /scr/user/kevin2002/TensorCat/uni-rumor/MDU/Defense_Engineering/outputs \
  -type f \( -iname '*probe*manifest*.json' -o -iname '*relevance*reference*.json' \
  -o -iname '*reproduction*.json' \) -print
```

Full training remains outside this package and is never triggered by either
mode.
