# Scientific Boundary

This branch implements an engineering contract and a deterministic mock
runtime. It does not implement or evaluate the frozen scientific model.

- No checkpoint is loaded, inspected, downloaded, or modified.
- No Validation or Test data is accessed.
- Mock SHA-256 scores and logits are non-scientific placeholders.
- Every mock result includes `MOCK_NON_SCIENTIFIC_OUTPUT`.
- Mock outputs are not accuracy, quality, calibration, or performance evidence.
- The only model classes are fake and real.
- NEI means engineering insufficient evidence/abstention, not a learned class.
- Visual/VLM observations are retained for display but excluded from frozen G1
  unless a later, separately authorized scientific change says otherwise.
