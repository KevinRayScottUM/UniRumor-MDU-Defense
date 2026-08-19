# PP-OCRv5 Integration

Task 04F adds pretrained OCR as upstream Defense engineering preprocessing. It
does not train or tune a model, access Validation/Test data, or change Frozen G1.

## Frozen external assets

| Role | Model ID | Frozen revision | Runtime tree SHA256 | Deployment metadata path |
|---|---|---|---|---|
| Detector | `PaddlePaddle/PP-OCRv5_server_det` | `ca867c897ecbca8873081573a802ad70d499cb94` | `a6e8aae048291ebff5d6b604ccda060ccf516ed82d5f8e5f4f4421e762395983` | `/scr/user/kevin2002/TensorCat/model_assets/PP-OCRv5_server_det` |
| Recognizer | `PaddlePaddle/PP-OCRv5_server_rec` | `b26c3587fda8da3c8ec0ce357214b4d661ff1558` | `248824aeede7ff94190ff2b82cce0679d89868713c749cc8cd3f6678006be259` | `/scr/user/kevin2002/TensorCat/model_assets/PP-OCRv5_server_rec` |

The paths above are deployment metadata. Runtime model directories are explicit
configuration and remain outside Git. The worker never downloads a model. Each
runtime-tree hash covers exactly these files, in filename order with deterministic
filename/byte separators:

- `config.json`
- `inference.json`
- `inference.pdiparams`
- `inference.yml`

Hugging Face `.cache` metadata is excluded. PP-OCRv5 is a pretrained upstream OCR
engine, not a project-trained checkpoint.

## ABI-isolated subprocess

Torch, Whisper, and Frozen G1 use cuDNN9, while PaddlePaddle 3.2.2 requires
`libcudnn.so.8`. The main Defense process therefore never imports Paddle or
PaddleOCR. It invokes `services.paddle_ocr_worker` with a subprocess argument
list and `shell=False`.

The child environment sets `OMP_NUM_THREADS=1`,
`DISABLE_MODEL_SOURCE_CHECK=True`, and prepends this isolated library directory
to the child's `LD_LIBRARY_PATH` only:

```text
/scr/user/kevin2002/TensorCat/runtime_libs/cudnn8-cu11/nvidia/cudnn/lib
```

The parent environment and cuDNN9 installation are not modified. Requests and
worker results are written only beneath the configured Defense cache root. A
nonzero exit, timeout, missing output, malformed JSON, model provenance mismatch,
or hash mismatch is a worker failure. A successful frame result containing zero
detections is valid empty OCR.

## Frame sampling and OCR units

The historical sampler uses eight frames by default. When a video has more than
eight frames, index `i` uses `(i + 1) / (n + 1)` across `frame_count - 1` with
Python `round`, clamping, and deterministic duplicate removal. Videos with at
most `n` frames expose every frame; `n=1` selects the middle frame. Frame time is
`frame_index / fps`. Extracted JPEGs stay under the configured cache root and the
source video is never modified.

The worker preserves every raw detection's text, real confidence, and polygon.
Its runtime bbox is the frozen flat schema:

```text
[min(x), min(y), max(x), max(y)]
```

Filtering defaults to confidence `>=0.5` and at least three Unicode alphanumeric
characters after NFKC normalization. Python Unicode semantics preserve valid CJK
text. Accepted detections are ordered top-to-bottom then left-to-right and joined
with single spaces into one OCR `RuntimeUnit` per frame. The unit carries the
mean real confidence, union bbox, deterministic frame/time identifiers, and all
accepted raw detection details (including polygons) in existing provenance
`details`. OCR units are eligible for Frozen G1.

Frame text is NFKC/casefold deduplicated. If more than six unique frame units
remain, candidates rank by mean confidence descending, raw detection count
descending, and frame rank ascending; the selected six are then restored to
chronological order. Raw worker artifacts are not truncated.

## Transcript and combined exposure

Raw Whisper output is unchanged and retained for UI, logging, provenance, and
debugging. At most 12 transcript exposure units are sent downstream. When raw
ASR has more than 12 segments, deterministic balanced contiguous groups cover
every source segment and the complete temporal range; source unit IDs remain in
provenance. The Task03 Golden Probe produced 31 raw transcript units, so it is a
contract regression case for full-history grouping, not model-performance
evidence.

Standard exposure order is transcript first (at most 12), then OCR (at most 6),
for a normal engineering maximum of 18. This reflects the frozen TRAIN-only
source-count audits (transcript `0..12`, OCR `0..6`) but is not a new scientific
model limit. Official Phase4A remains `max_units=24`, `max_length=256`, and
class-wise max pooling over all exposed eligible units; Top-5 remains explanation
only.

ASR-only and OCR-only evidence may each reach the existing `FrozenG1Runner`.
When both are empty, Frozen G1 is not called and the existing engineering
`NOT_RUN` / insufficient-evidence / `NEI` schema is returned. NEI is not a third
model class. Visual observations remain excluded from Frozen G1, and real
CLIP/VLM integration is not part of this task.
