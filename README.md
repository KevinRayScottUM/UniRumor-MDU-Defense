# UniRumor MDU — Defense Engineering System

This branch contains the engineering-oriented graduation defense system.

Target runtime pipeline:

Video + Claim
-> video/audio preprocessing
-> ASR
-> OCR
-> CLIP/SigLIP frame retrieval
-> VLM visual observations
-> MDU candidate construction
-> Frozen G1 DeBERTa inference
-> sample-level verdict
-> Top-k explanatory units
-> Web application visualization

The frozen scientific model is used as a read-only decision engine.
Engineering development must not modify the frozen official Test results.
