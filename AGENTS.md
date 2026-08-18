# UniRumor MDU Defense Runtime Rules

## Scope
- Work only in this repository and the current branch.
- Keep runtime writes under configured `cache/` and `outputs/` roots.
- Do not read or write Validation/Test data.
- Do not load, inspect, download, or alter real checkpoints.
- Do not commit, push, merge, or modify `main` unless explicitly requested.

## Frozen Scientific Contract
- Backbone identity: `microsoft/deberta-v3-base`.
- Maximum evaluated eligible units: 24.
- Maximum sequence length: 256.
- Sample pooling: class-wise maximum over every evaluated eligible unit.
- Labels: `fake=0`, `real=1`.
- Top-k: 5, explanation-only.
- Selection scores never determine the sample prediction pool.
- Text, transcript, and OCR units may be eligible for frozen G1.
- Visual/VLM observations default to `eligible_for_frozen_g1=False`.
- NEI is an engineering abstention display state, never a model class.

## Runtime Boundary
- The runtime skeleton is deterministic and standard-library-only.
- Mock scores and logits must use SHA-256, never Python `hash()`.
- Mock outputs must carry `MOCK_NON_SCIENTIFIC_OUTPUT`.
- Do not present mock output as research evidence or model performance.
- Do not add Torch, Transformers, ASR, OCR, vision, VLM, CUDA, video,
  web-server, or UI dependencies in this milestone.

## Verification
- Use standard-library `unittest`.
- Run `python3 -m unittest discover -s tests -v`.
- Run `python3 -m app.mock_demo`.
- Run `git diff --check`.
