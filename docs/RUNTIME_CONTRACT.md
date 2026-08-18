# Runtime Contract

The runtime accepts a `VerificationRequest`, creates a deterministic session,
produces typed `RuntimeUnit` objects, and returns a `VerificationResult`. Every
schema supports deterministic `to_dict`/`from_dict` JSON round-trips.

## Frozen values

| Setting | Value |
|---|---|
| Backbone identity | `microsoft/deberta-v3-base` |
| Maximum evaluated units | 24 eligible units |
| Maximum sequence length | 256 |
| Pooling | class-wise maximum |
| Labels | `fake=0`, `real=1` |
| Explanation units | top 5 by selection score |

Prediction uses the class-wise maximum logits over **all evaluated eligible
units**. Selection scores rank explanations only and never filter the prediction
pool. Text, transcript, and OCR may be eligible. Visual observations are
ineligible by default.

If no eligible unit exists, G1 is not run: `model_verdict=not_run`,
`evidence_status=insufficient`, and `display_verdict=NEI`. NEI has no logit or
probability because it is not a model class.
