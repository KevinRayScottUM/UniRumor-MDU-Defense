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
  |-- PyAV 16 kHz mono waveform
  |     -> frozen external openai/whisper-large-v3-turbo
  |     -> raw transcript RuntimeUnits
  |     -> balanced full-range transcript exposure (at most 12)
  |
  |-- historical deterministic 8-frame sampler
        -> isolated PP-OCRv5 subprocess
        -> raw OCR detections
        -> frame-level OCR RuntimeUnits (at most 6)
  |
  -> transcript exposure first, then OCR exposure (normally at most 18)
  -> optional existing FrozenG1Runner
  -> external Phase4A Frozen G1
  -> VerificationResult
```

The real path integrates video audio decoding, pretrained Whisper ASR, isolated
pretrained PP-OCRv5, and the existing external Frozen G1 bridge. Full raw ASR
segments and raw OCR artifacts remain available independently from the bounded
G1 exposure. The normal engineering exposure of 18 is not a new model limit;
official Phase4A `max_units=24` remains unchanged. Real CLIP/VLM integration is
**not yet integrated**, and visual observations remain G1-ineligible.

Across both paths, selection scores order explanatory units while fake/real
logits from every evaluated eligible unit determine the sample verdict. The mock
orchestration layer records explicit ordered stage states. Runtime cache entries,
JSONL logs, and result JSON are restricted to configured `cache/` and `outputs/`
roots.
