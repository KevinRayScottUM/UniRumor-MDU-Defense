# Frozen G1 Phase4A Integration

This repository contains only a thin bridge to the external, read-only Phase4A
inference CLI on DICC. It does not contain, inspect, load, download, train, or
copy the model, checkpoint, tokenizer, or dataset assets.

## Frozen external contract

- Variant: `G1_text_ocr`
- Backbone identity: `microsoft/deberta-v3-base`
- Checkpoint SHA-256: `b694f2d4bb5ba6f72dd8a001bd984d46853546f2a85858a812f2496af1f1a0b9`
- Phase4A contract: `1.0.0`
- Pooling: class-wise maximum over all model-exposed eligible units
- Limits: 24 exposed units, sequence length 256
- Labels: `fake=0`, `real=1`
- Top-k: at most 5 units, for explanation only

`adapters/phase4a_request_adapter.py` maps eligible text, transcript, and OCR
`RuntimeUnit` objects to `candidate_units` without local truncation. Visual
observations are never submitted. `adapters/phase4a_response_adapter.py`
strictly validates the frozen identity, settings, output IDs, and all-unit
class-wise maxima before any scores are attached to runtime units.

`services/frozen_g1_runner.py` writes the one-record request JSONL beneath the
configured Defense cache root, invokes the external CLI with an argument list
and `shell=False`, and reads the prediction JSONL beneath the configured Defense
output root. The external project root, Python executable, inference script,
config, device, timeout, cache root, and output root are runtime settings. No
machine-specific paths are embedded in the bridge.

The command has this shape:

```text
<PYTHON> -u <PHASE4A_INFER> --config <PHASE4A_CONFIG> \
  --project-root <UNIRUMOR_ROOT> --input <REQUEST_JSONL> \
  --output <PREDICTION_JSONL> --device <DEVICE>
```

The bridge deliberately does not pass `--drop-unsupported-visual`.

If there are no eligible units, the subprocess is skipped and the result is
the engineering abstention state: `not_run`, `insufficient`, and `NEI`, with no
logits or probabilities. A successful validated external prediction is marked
`sufficient`; it never receives the mock-output warning.

Local tests use a static prediction fixture and a stubbed subprocess. Do not run
real Phase4A inference on Mac, and do not use this bridge to access Validation or
Test data.
