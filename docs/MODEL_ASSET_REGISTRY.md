# Model Asset Registry

This is a metadata-only inventory. `TBD` means the value is unknown and has not
been inspected, loaded, downloaded, or resolved by this runtime foundation.

| logical_name | model_name | model_version | source | local_path | device | dtype | sha256 | required_vram | runtime_status |
|---|---|---|---|---|---|---|---|---|---|
| g1_checkpoint | microsoft/deberta-v3-base | TBD | TBD | TBD | TBD | TBD | TBD | TBD | identity_only_not_loaded |
| g1_tokenizer | microsoft/deberta-v3-base tokenizer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | identity_only_not_loaded |
| ocr | PP-OCRv5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not_integrated |
| visual_retrieval | CLIP/SigLIP2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not_integrated |
| vlm | Qwen2.5-VL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not_integrated |
| asr | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not_integrated |

The runtime must not resolve paths, inspect hashes, access checkpoints, or import
ML frameworks. `checkpoint_sha256` therefore remains `null` in mock results.
