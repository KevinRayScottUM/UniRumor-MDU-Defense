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
units.

## Unified Production Runtime

```text
ProductionRuntimeConfig
  -> ProductionRuntimeFactory
  -> ProductionRuntimeServices
  -> ProductionRuntime
  -> VideoMultimodalRunner
```

`ProductionRuntime` construction is side-effect free. Its first `start()` runs
factory preflight and builds the service graph once; later starts and requests
reuse that exact graph while real models remain lazy until execution demands
them. The caller supplies a safe session ID and the exact claim is preserved.
The source video is resolved and validated as a read-only regular file before
startup, then execution delegates only to `VideoMultimodalRunner`.

Runtime failures propagate from this lifecycle layer to the public-safe
operational failure boundary described below. The Frozen G1 contract remains
unchanged, and supplemental visual observations remain excluded from Frozen G1
candidate units.

## Structural Evidence Sufficiency

`EvidenceSufficiencyPolicy` audits a successfully completed
`VideoMultimodalResult` and emits a separate
`EvidenceSufficiencyAssessment`; it does not modify the verification result.
Sufficiency is structural, never probability- or logit-threshold based:
G1-eligible exposure plus a completed Frozen G1 binary result is sufficient,
while no G1-eligible exposure remains engineering NEI/insufficient. No score or
minimum-unit threshold is introduced.

Visual observations remain supplemental and can never independently make
evidence sufficient. Operational and model failures remain outside this policy
and propagate to the execution boundary. Task06E packages the assessment into
an API-ready result.

## API-Ready Production Result

```text
VideoMultimodalResult + EvidenceSufficiencyAssessment
  -> ProductionResultBuilder
  -> ProductionResult
```

This presentation and serialization layer does not change predictions;
logits and probabilities are preserved exactly. G1 exposure remains separate
from supplemental visual evidence, while ordered Top-k explanation IDs refer
back to the G1 exposure list and never imply a Top-k prediction basis.

The public contract intentionally omits local filesystem paths, raw provenance
details, and raw internal warning strings. Successful engineering NEI results
serialize normally. The Task06G production CLI consumes this contract, and
Task07 FastAPI/Gradio surfaces may later expose the same contract.

## Operational Failure Boundary

Successful Fake/Real and successful NOT_RUN/NEI results both have execution
status `success`. A runtime or result-packaging exception has execution status
`failure`; it is neither evidence insufficiency nor a Frozen G1 model class, and
the failure path never fabricates NEI or another verdict.

The deliberately coarse public failure payload exposes only a stable stage and
code, the exception type name, and a fixed message. Raw exception messages,
tracebacks, filesystem paths, subprocess stderr, and internal warnings are not
public. Task06G uses `ProductionExecutionOutcome` for CLI JSON and exit codes,
while Task07 may later map the same contract to HTTP semantics.

## Production CLI

The official command-line entry point composes the existing production
execution service without adding another inference or result pipeline:

```bash
python -m app.production_cli \
    --config /path/to/production_runtime.json \
    --session-id demo-001 \
    --claim "..." \
    --video /path/to/video.mp4
```

Standard output is exactly one compact `ProductionExecutionOutcome` JSON
document. Exit code `0` means successful execution, including Fake, Real, and
normal NOT_RUN/NEI results. Exit code `1` means Task06F returned a public-safe
operational failure outcome; an operational failure never becomes NEI. Exit
code `2` means CLI usage, configuration, or service initialization failed.

The CLI does not expose raw exceptions or local paths, and it does not alter
scientific parameters. Deployment paths, devices, and model settings remain in
the production configuration. Task07 FastAPI/Gradio surfaces will reuse
`ProductionExecutionService` directly rather than shelling out to this CLI.
