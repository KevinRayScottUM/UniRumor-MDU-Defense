# Video + ASR Integration

This thin integration decodes a video's audio stream with PyAV, resamples it in
memory to a contiguous 16 kHz mono float32 waveform, transcribes it with the
configured local Whisper asset, and maps ordered transcript segments to Frozen
G1-compatible `RuntimeUnit` objects. It does not create temporary WAV files or
invoke the ffmpeg/ffprobe command-line tools.

## External ASR asset

- Model: `openai/whisper-large-v3-turbo`
- Frozen revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- `model.safetensors` SHA256:
  `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1`
- Deployment metadata path:
  `/scr/user/kevin2002/TensorCat/model_assets/openai__whisper-large-v3-turbo`

The path above is documentation only. Callers must provide `--model-path`; the
core service has no DICC path default. Transformers receives
`local_files_only=True`, so it cannot implicitly download the model. Optional
strict verification hashes the configured local `model.safetensors` before any
model or processor is loaded.

## Runtime flow

```text
video -> PyAV -> 16 kHz mono waveform -> local Whisper
      -> ordered ASR segments -> transcript RuntimeUnits
      -> optional existing FrozenG1Runner
```

Whisper inference requests timestamped transcription without setting a language,
preserving detected multilingual text. Blank chunks are removed. Negative,
reversed, non-finite, and non-monotonic timestamps are rejected. Transcript
units use deterministic session-local IDs (`asr_0000`, `asr_0001`, ...), carry
no confidence, selection score, or logits, and are eligible for Frozen G1.

When ASR yields no valid transcript units, the integration returns an existing
`VerificationResult` with `model_verdict=not_run`, insufficient evidence, and
the `NEI` display state. It does not call Frozen G1 in this case.

## Demo

The demo performs real local inference and therefore is intended for the DICC
runtime, not dependency-light Mac tests:

```bash
python3 -m app.video_asr_demo \
  --session-id example \
  --claim "Example claim" \
  --video /path/to/video.mp4 \
  --model-path /configured/external/whisper/path \
  --device cuda:0 \
  --verify-asset-sha256
```

`--max-duration` is optional and deployment-configurable; no Golden Probe
duration is embedded in the decoder.
