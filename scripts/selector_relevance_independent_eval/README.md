# Step 2.6R-3B3 independent selector evaluation

Implementation revision: `step2.6r-3b3-v1`.

This isolated package implements two deliberately separate modes:

- `--preflight` verifies the frozen 3B1 cohort and preregistration, the frozen
  3B2 relevance gold, the approved Stage-A invariance result, the closed
  neutral/training artifacts, seed 42, the selector SHA, the Frozen G1 SHA, and
  the Phase4A configuration. It does not instantiate the runtime, import
  Torch for execution, load checkpoint tensors, or score a selector.
- `--one-shot-evaluate` requires and re-verifies the atomic preflight packet
  before constructing `DICCEvaluationRuntime`. It scores the original and the
  preregistered seed-42 selector once, preserves all 30 cases, excludes only
  the two zero-DIRECT cases from macro denominators, and atomically freezes a
  scientifically valid PASS or FAIL.

Only `scripts.selector_relevance_gate.runtime` and
`scripts.selector_relevance_gate.metrics` are reused. The historical
`run_heldout_gate`, `load_heldout_references`, `heldout_loader.py`, CPAC gate,
and six-case evidence are not used.

The CLI intentionally has no seed, threshold, overwrite, retry, model-select,
Validation, or Test option. Existing output directories fail closed. Exit
codes are 0 for preflight or a valid scientific PASS, 1 for a valid scientific
FAIL, and 2 for invalid input/protocol/runtime execution.

On DICC, `--project-root` is the UniRumor repository root consumed by the
authoritative Phase3/Phase4A source loaders and by relative checkpoint paths:

```sh
ROOT=/scr/user/kevin2002/TensorCat/uni-rumor
DEFENSE="$ROOT/MDU/Defense_Engineering"
```

Both modes must pass `--project-root "$ROOT"`, never `--project-root
"$DEFENSE"`. The Phase4A configuration, closed 1D neutral directory, and
closed Step 2.6R-2 training directory remain explicit required CLI inputs; the
implementation does not guess dated DICC output paths.

The score-free preflight does not require a CUDA determinism environment:

```sh
PYTHONPATH="$DEFENSE" "$PY" \
  -m scripts.selector_relevance_independent_eval.run_evaluation \
  --preflight \
  --project-root "$ROOT" \
  --cohort-dir "$COHORT_DIR" \
  --final-gold-dir "$FINAL_GOLD_DIR" \
  --stage-a-invariance-report "$STAGE_A_INVARIANCE_REPORT" \
  --phase4a-config "$PHASE4A_CONFIG" \
  --neutral-dir "$NEUTRAL_DIR" \
  --training-dir "$TRAINING_DIR" \
  --output-dir "$PREFLIGHT_DIR"
```

The CUDA one-shot invocation must set the deterministic cuBLAS workspace
contract required by `DICCEvaluationRuntime`:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$DEFENSE" "$PY" \
  -m scripts.selector_relevance_independent_eval.run_evaluation \
  --one-shot-evaluate \
  --project-root "$ROOT" \
  --cohort-dir "$COHORT_DIR" \
  --final-gold-dir "$FINAL_GOLD_DIR" \
  --approved-preflight-report \
    "$PREFLIGHT_DIR/one_shot_preflight_report.json" \
  --stage-a-invariance-report "$STAGE_A_INVARIANCE_REPORT" \
  --phase4a-config "$PHASE4A_CONFIG" \
  --neutral-dir "$NEUTRAL_DIR" \
  --training-dir "$TRAINING_DIR" \
  --device cuda:0 \
  --output-dir "$EVALUATION_DIR"
```
