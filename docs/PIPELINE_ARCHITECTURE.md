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
  |-- historical CLIP12 candidate extraction
        -> local frozen SigLIP2 claim retrieval
        -> relevance Top-4 restored chronologically
        -> local frozen claim-blind Qwen2.5-VL observer
        -> grounded visual_observation RuntimeUnits (G1-ineligible)
  |
  -> transcript exposure first, then OCR exposure (normally at most 18)
  -> append supplemental visual observations
  -> existing FrozenG1Runner filters visual units before Phase4A
  -> external Phase4A Frozen G1
  -> VerificationResult
```

The real path integrates video audio decoding, pretrained Whisper ASR, isolated
pretrained PP-OCRv5, and the existing external Frozen G1 bridge. Full raw ASR
segments and raw OCR artifacts remain available independently from the bounded
G1 exposure. The normal engineering exposure of 18 is not a new model limit;
official Phase4A `max_units=24` remains unchanged. The real visual path is
supplemental: SigLIP ranks frames, Qwen observes them without the claim, and all
visual observations remain G1-ineligible. Visual-only evidence yields engineering
NEI/NOT_RUN rather than a fake/real prediction.

Across both paths, selection scores order explanatory units while fake/real
logits from every evaluated eligible unit determine the sample verdict. The mock
orchestration layer records explicit ordered stage states. Runtime cache entries,
JSONL logs, and result JSON are restricted to configured `cache/` and `outputs/`
roots.

## Runtime Configuration Contracts

`pipeline.RuntimeConfig` remains the deterministic mock-runtime configuration.
`ProductionRuntimeConfig` is a separate deployment-only contract for local
paths, devices, data types, and subprocess timeouts used by the existing real
services. It cannot change Frozen G1 scientific constants or authorize visual
observations for G1.

Production configuration parsing is portable and side-effect free: it neither
loads models nor requires external assets to exist.

## Production Service Factory

```text
ProductionRuntimeConfig
  -> ProductionRuntimeFactory
  -> existing real service graph
  -> VideoMultimodalRunner
```

Factory construction is lazy: models load only when later runtime execution
demands them. Its optional preflight checks filesystem deployment prerequisites
only; it does not load or hash model weights, and the existing services retain
ownership of frozen-asset verification. The Frozen G1 scientific contract is
unchanged, and visual observations remain excluded from Frozen G1 candidate
units. Task06C will manage runtime lifecycle and execution-wrapper behavior.
