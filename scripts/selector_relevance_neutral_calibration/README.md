# Step 2.6R-1D Modality-Neutral Calibration Revision

This package derives a new immutable calibration artifact from the closed Step
2.6R-1A v2 artifact. The only transformed field is `claim`; OCR and Transcript
examples both become:

```text
The relevant content states "<ANCHOR_TEXT>".
```

It verifies source hashes and registered invariants before writing. It does not
load a model/checkpoint, inspect selector or veracity outputs, access Formal
Validation/Test, or train.

```bash
python -m scripts.selector_relevance_neutral_calibration.run_build_neutral \
  --source-dir /absolute/path/to/02_dataset_build_provenance_corrected \
  --output-dir /absolute/path/to/01_neutral_revision
```
