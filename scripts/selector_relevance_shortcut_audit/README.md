# Step 2.6R-1C Template / Modality Shortcut Audit

This package performs a read-only audit of the locked Step 2.6R-1A v2
calibration artifact and the actual DICC Frozen G1 source-level claim/unit
encoding contract. It does not import or execute Frozen G1, load a checkpoint
or tokenizer, inspect selector/veracity outputs, train, or rewrite calibration
JSONL files.

The audit verifies calibration SHA sidecars and build-report hashes, computes
claim-only template/modality leakage from the data, statically identifies the
authoritative tokenizer pair call with Python AST inspection, and emits a
diagnostic-only neutral/swapped template manifest.

Run on DICC:

```bash
python -m scripts.selector_relevance_shortcut_audit.run_audit \
  --project-root /absolute/path/to/uni-rumor \
  --calibration-dir /absolute/path/to/selector_relevance_calibration_v2/02_dataset_build_provenance_corrected \
  --output-dir /absolute/path/to/outputs/selector_relevance_shortcut_audit_v1
```

The template-swapped control is diagnostic only and must never be presented as
literal production evidence.
