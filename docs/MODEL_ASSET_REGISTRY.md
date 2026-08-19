# Model Asset Registry

This is a metadata-only inventory. `TBD` means the value is unknown and has not
been inspected, loaded, downloaded, or resolved by this runtime foundation.

| logical_name | model_name | model_version | source | local_path | device | dtype | sha256 | required_vram | runtime_status |
|---|---|---|---|---|---|---|---|---|---|
| g1_checkpoint | microsoft/deberta-v3-base / G1_text_ocr | Phase4A contract 1.0.0 | external DICC runtime | runtime-configured external path | runtime-configured | TBD | b694f2d4bb5ba6f72dd8a001bd984d46853546f2a85858a812f2496af1f1a0b9 | TBD | external_bridge_only_not_loaded |
| g1_tokenizer | microsoft/deberta-v3-base tokenizer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | identity_only_not_loaded |
| ocr_detector | PaddlePaddle/PP-OCRv5_server_det | ca867c897ecbca8873081573a802ad70d499cb94 | Hugging Face frozen revision; external DICC asset | /scr/user/kevin2002/TensorCat/model_assets/PP-OCRv5_server_det (deployment metadata only; runtime-configurable) | isolated Paddle CUDA worker | model runtime default | a6e8aae048291ebff5d6b604ccda060ccf516ed82d5f8e5f4f4421e762395983 (runtime tree) | TBD | verified external asset |
| ocr_recognizer | PaddlePaddle/PP-OCRv5_server_rec | b26c3587fda8da3c8ec0ce357214b4d661ff1558 | Hugging Face frozen revision; external DICC asset | /scr/user/kevin2002/TensorCat/model_assets/PP-OCRv5_server_rec (deployment metadata only; runtime-configurable) | isolated Paddle CUDA worker | model runtime default | 248824aeede7ff94190ff2b82cce0679d89868713c749cc8cd3f6678006be259 (runtime tree) | TBD | verified external asset |
| visual_retrieval | CLIP/SigLIP2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not_integrated |
| vlm | Qwen2.5-VL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | not_integrated |
| asr | openai/whisper-large-v3-turbo | 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 | Hugging Face frozen revision; external DICC asset | /scr/user/kevin2002/TensorCat/model_assets/openai__whisper-large-v3-turbo (deployment metadata only; runtime-configurable) | runtime-configured CUDA | float16 | 542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1 (model.safetensors) | TBD | verified external asset |

The deterministic mock runtime does not resolve paths, inspect hashes, access
checkpoints, or import ML frameworks. `checkpoint_sha256` therefore remains
`null` in mock results. The Frozen G1 bridge validates the externally reported
checkpoint SHA-256 above; the checkpoint itself remains external and is never
inspected by this repository. The optional Video + ASR integration accepts its
Whisper path only through runtime configuration, loads with
`local_files_only=True`, and can verify `model.safetensors` before inference.
The PP-OCRv5 worker accepts only explicit local detector/recognizer directories
and verifies the exact runtime tree (`config.json`, `inference.json`,
`inference.pdiparams`, and `inference.yml`) without hashing Hugging Face cache
metadata. PP-OCRv5 is a pretrained upstream OCR engine, not a project-trained
checkpoint.
