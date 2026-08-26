# Step 2.6R-1A Direct-Relevance Dataset Builder

This package builds deterministic, Train-only direct claim-unit relevance data
for a future explanation selector. It does not train a model, load a checkpoint,
run Frozen G1 inference, or change production behavior.

The builder is deliberately score-blind and veracity-label-blind. It constructs
synthetic direct-grounding claims from quality-filtered OCR and Transcript units
only after the actual Phase4A `normalize_request` policy has produced the
model-exposed candidate pool. There is no approximate local exposure fallback.

Each eligible underlying case contributes exactly two OCR and two Transcript
examples. The five Step 2.5B cases and the CPAC strict-audit case remain held
out. Formal Validation and Formal Test are not opened.

## DICC execution

Run from the Defense Engineering repository root with the existing DICC Python
environment. The CLI verifies the Phase3A Train-lock report, the locked Train
source SHA-256, the Step 2.5B held-out manifest, and the Phase4A configuration
before writing only beneath `--output-dir`.

```bash
python -m scripts.selector_relevance_calibration.run_build_dataset \
  --project-root /absolute/path/to/uni-rumor \
  --phase3a-train-lock-report /absolute/path/to/clip12p3a_train_lock_report.json \
  --phase4a-config /absolute/path/to/clip12_phase4a_frozen_g1_inference_handoff.json \
  --step25b-selected-manifest /absolute/path/to/train_reference_selection_manifest.json \
  --heldout-case GroundLie360:13025004 \
  --output-dir /absolute/path/to/outputs/selector_relevance_calibration_v1/01_dataset_build
```

Mac tests inject a controlled exposure adapter. They do not claim real DICC
integration or dataset counts.
