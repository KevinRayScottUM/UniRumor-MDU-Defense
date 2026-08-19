# UniRumor MDU Defense Runtime Rules

## Scope
- Work only in this repository and the current branch.
- Keep runtime writes under configured `cache/` and `outputs/` roots.
- Do not read or write Validation/Test data.
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

## Runtime Paths

### A. Deterministic Mock Regression Path
- Keep mock behavior deterministic and standard-library-oriented.
- Mock scores and logits must use SHA-256, never Python `hash()`.
- Mock outputs must carry `MOCK_NON_SCIENTIFIC_OUTPUT`.
- Never present mock output as scientific evidence, research performance, model
  quality, accuracy, or calibration.

### B. Real Engineering Integrations
- The repository includes an external Frozen G1 Phase4A inference bridge and an
  external pretrained Whisper ASR integration.
- PP-OCRv5 runs only in its dedicated subprocess worker because Paddle requires
  the isolated cuDNN8 runtime; the main Defense process must never import Paddle
  or PaddleOCR or mutate its own `LD_LIBRARY_PATH`.
- The real visual path uses local-only SigLIP2 retrieval followed by a claim-blind
  Qwen2.5-VL observer. Its visual observations are supplemental UI evidence and
  must always remain ineligible for Frozen G1.
- Future vision changes are allowed only when explicitly scoped.
- Keep every external model asset outside Git and access it only through
  configurable local or deployment paths.
- Never implicitly download external model assets.
- Do not train or tune models unless explicitly authorized.
- Never access Validation/Test data for engineering tuning or integration work.
- Never alter the Frozen G1 scientific constants, contract, or evaluation
  boundaries through an engineering integration.

## Verification
- Use standard-library `unittest`.
- Run `python3 -m unittest discover -s tests -v`.
- Run `python3 -m app.mock_demo`.
- Run `git diff --check`.
