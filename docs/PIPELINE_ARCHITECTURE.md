# Pipeline Architecture

## Mock Regression Path

```text
VerificationRequest
  -> deterministic session
  -> mock ASR transcript units
  -> mock OCR units
  -> mock visual retrieval
  -> mock VLM visual_observation units (G1-ineligible)
  -> RuntimeUnit pool
  -> first 24 eligible units -> deterministic mock G1
  -> class-wise max over all evaluated eligible units
  -> VerificationResult + top-5 explanatory units
```

This path exercises mock ASR, OCR, visual retrieval, VLM, and G1 behavior. It is
deterministic regression infrastructure, not scientific inference evidence.

## Current Real Engineering Path

```text
video
  -> PyAV 16 kHz mono waveform
  -> frozen external openai/whisper-large-v3-turbo
  -> ordered transcript RuntimeUnits
  -> optional existing FrozenG1Runner
  -> external Phase4A Frozen G1
  -> VerificationResult
```

The real path currently integrates video audio decoding, pretrained Whisper ASR,
and the existing external Frozen G1 bridge. Real OCR integration and real
CLIP/VLM integration are **not yet integrated**.

Across both paths, selection scores order explanatory units while fake/real
logits from every evaluated eligible unit determine the sample verdict. The mock
orchestration layer records explicit ordered stage states. Runtime cache entries,
JSONL logs, and result JSON are restricted to configured `cache/` and `outputs/`
roots.
