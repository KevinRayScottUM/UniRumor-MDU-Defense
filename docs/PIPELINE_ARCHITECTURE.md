# Pipeline Architecture

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

The runtime keeps two independent concepts: selection scores order explanatory
units, while fake/real logits from every evaluated eligible unit determine the
sample verdict. The orchestration layer records explicit ordered stage states.
Cache entries, JSONL logs, and result JSON are restricted to configured local
`cache/` and `outputs/` roots.
