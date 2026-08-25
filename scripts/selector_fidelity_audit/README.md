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
