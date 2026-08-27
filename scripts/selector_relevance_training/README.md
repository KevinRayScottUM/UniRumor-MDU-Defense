# Step 2.6R-2 Direct-Relevance Selector-Only Calibration

This isolated DICC-only trainer updates exactly `selection_head.weight` and
`selection_head.bias`. It reuses the authoritative Phase3A `collator`, freezes
the encoder and veracity head, caches deterministic encoder representations in
evaluation mode, and optimizes explicit `relevance_target` values with
`BCEWithLogitsLoss`.

The source is the closed Step 2.6R-1D neutral artifact. Source JSONLs,
manifest, Frozen G1 checkpoint, architecture constants, and safety boundaries
are verified before training. Formal Validation, Formal Test, Step 2.5B and
CPAC held-out content are never inputs.

`--smoke` runs seed 42 for one epoch on eight Train and four Dev examples. It
does not trigger `--full`. Full training requires an explicitly supplied,
previously inspected PASS smoke report and runs seeds 42, 43 and 44 for at
most ten epochs.

The original Frozen G1 checkpoint is never overwritten. Persisted `.pt`
artifacts contain only the selection-head state plus registered provenance and
training metadata; encoder and veracity-head tensors are excluded.

Run `python -m scripts.selector_relevance_training.run_train --help` for the
explicit DICC paths and execution mode.
