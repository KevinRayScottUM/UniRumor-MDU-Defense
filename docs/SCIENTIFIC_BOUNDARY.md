# Scientific Boundary

This repository contains five distinct runtime capabilities:

- A deterministic mock regression runtime. Its SHA-256 scores and logits are
  non-scientific placeholders, every mock result includes
  `MOCK_NON_SCIENTIFIC_OUTPUT`, and mock outputs are never accuracy, quality,
  calibration, performance, or other scientific evidence.
- A real external Frozen G1 Phase4A inference bridge. It invokes the frozen
  scientific decision engine without changing its contract or checkpoint.
- A real pretrained Whisper ASR engineering service. It converts video audio
  into transcript evidence upstream of Frozen G1.
- A real pretrained PP-OCRv5 engineering service. It converts sampled video
  frames into frame-level OCR evidence in an ABI-isolated worker subprocess.
- A real supplemental visual service. Frozen SigLIP2 ranks candidate frames by
  claim relevance, then frozen Qwen2.5-VL observes only the selected frames under
  a claim-blind, no-OCR, no-veracity prompt and grounded automatic filter.

## Frozen G1 Boundary

- Frozen G1 remains a binary `fake`/`real` model.
- `max_units=24` and `max_length=256` remain fixed.
- Sample pooling remains the class-wise maximum over every evaluated eligible
  unit.
- Top-5 remains explanation-only; selection scores never determine the sample
  prediction pool.
- Visual/VLM observations remain excluded from Frozen G1.
- NEI remains an engineering insufficient-evidence/abstention display state,
  never a learned model class.

Whisper ASR and PP-OCRv5 are upstream engineering preprocessing. They are not
part of the Frozen G1 research performance claim and do not change the scientific
model. The transcript/OCR exposure policy (normally at most 12 plus 6) is an
engineering composition policy, not a replacement for the official Phase4A
`max_units=24` boundary. These integrations perform no model training or tuning
and must never access Validation/Test data for engineering tuning. External
assets remain outside Git.

Visual observations are engineering UI/evidence artifacts, not Frozen G1 inputs
or visual-veracity predictions. Adding them must leave the Phase4A candidate list
identical to the corresponding text/transcript/OCR-only request. Visual-only
evidence cannot produce a binary model verdict and returns engineering NEI.
