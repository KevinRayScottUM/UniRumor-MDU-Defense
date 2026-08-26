# Frozen G1 Selector Fidelity Audit

This directory is audit-only. It does not change Frozen G1, Phase4A, Top-k,
candidate exposure, selection scores, veracity logits, or production behavior.

The checkout contains only the external Phase4A bridge. Real execution is
therefore DICC-only and requires:

1. an exported public CPAC result containing the exact ordered 18-unit
   `g1_exposure_units` pool;
2. the existing DICC production runtime configuration;
3. completed human relevance annotations for the three transcript probes in
   `probe_definitions.json`.

Do not point either input at Validation or Test. The runner rejects paths with
those exact path components.

After human annotation, run from the Defense Engineering repository root:

```bash
/scr/user/kevin2002/TensorCat/.venv310/bin/python -m \
  scripts.selector_fidelity_audit.run \
  --candidate-pool /path/to/exported-cpac-public-result.json \
  --runtime-config /path/to/production-runtime.json \
  --output-dir outputs/selector_fidelity_audit
```

The runner preserves candidate order, creates fresh `RuntimeUnit` objects for
every claim, delegates scoring to the existing `FrozenG1Runner`, and writes the
four required audit artifacts. Visual observations are rejected from the
Frozen G1 candidate input.

## Step 2.5B cross-case confirmation

`cross_case.py` extends the same loader, runner, rank derivation, authoritative
Top-k membership, candidate hashing, and relevance metrics. It does not replace
the single-case audit.

Cross-case discovery recognizes two explicit reference contracts:

- `PUBLIC_RESULT`: the existing production-style result containing ordered
  `g1_exposure_units` and public Top-k explanation IDs;
- `NATIVE_PHASE4A`: a native prediction record containing `claim`, ordered
  `unit_outputs`, and `top_k_selection_units`.

For native records, score-blind discovery reconstructs candidates exclusively
from `unit_id`, `unit_type`, `modality`, and `text`. Selection scores, unit
veracity logits, Top-k contents, sample logits, probabilities, and prediction
are first consulted during the reproduction stage, after
`selected_case_manifest.json` has been written.

Discovery is score-blind: cases are deduplicated by the model-input candidate
hash and by canonical underlying identity. Canonicalization removes only exact
colon-delimited split tokens (`train`, `test`, `validation`, `val`) and retains
dataset/example identity. One lexically first independent case is selected per
dataset; remaining slots use stable metadata order. The scanner prunes exact
`Validation`/`Test` and `selector_fidelity_audit` path components, excludes the
current output directory, and rejects forbidden input roots before opening
files. Every selected case must replay its original source claim within the
declared `1e-6` reproduction tolerance before automatic probes are made.

The balanced direct-grounding manifest and its SHA-256 are written before any
new probe scoring. Real execution remains DICC-only:

```bash
cd /absolute/path/to/UniRumor-MDU-Defense
/scr/user/kevin2002/TensorCat/.venv310/bin/python -m \
  scripts.selector_fidelity_audit.run_cross_case \
  --discovery-root PUBLIC_QA=/absolute/non-Test/public-qa/artifacts \
  --discovery-root TRAIN_DERIVED=/absolute/non-Test/training-derived/artifacts \
  --runtime-config /absolute/non-Test/path/production-runtime.json \
  --output-dir outputs/selector_fidelity_audit/cross_case
```

Do not substitute Validation or Test paths. If fewer than four independent
pools pass reproduction, the output classification is `INCONCLUSIVE`. The
cross-case outputs do not overwrite the existing single-case report.
