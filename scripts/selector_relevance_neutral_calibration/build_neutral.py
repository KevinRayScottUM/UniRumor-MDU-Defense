"""Build a claim-only modality-neutral revision of the closed v2 artifact.

The implementation is deliberately standard-library-only and never imports or
executes Frozen G1, Phase3A, Phase4A, a tokenizer, or a model runtime.  It reads
the closed calibration artifact, changes only ``claim``, proves every registered
invariant, and writes a new immutable derived artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


IMPLEMENTATION_REVISION = "step2.6r-1d-v1"
SOURCE_IMPLEMENTATION_REVISION = "step2.6r-1a-v2"
OCR = "OCR"
TRANSCRIPT = "TRANSCRIPT"
OCR_PREFIX = 'The on-screen text reads "'
TRANSCRIPT_PREFIX = 'The speaker says "'
NEUTRAL_PREFIX = 'The relevant content states "'
CLAIM_SUFFIX = '".'
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESTRICTED_PATH_PARTS = {"validation", "test", "formalvalidation", "formaltest"}
_REQUIRED_SOURCE_FILES = (
    "build_report.json",
    "calibration_train.jsonl",
    "calibration_train.sha256",
    "calibration_dev.jsonl",
    "calibration_dev.sha256",
    "calibration_split_manifest.json",
    "eligible_case_manifest.json",
)
_OPTIONAL_SOURCE_SIDECARS = (
    "calibration_split_manifest.sha256",
    "eligible_case_manifest.sha256",
)
_FORBIDDEN_SCIENTIFIC_KEYS = frozenset(
    {
        "label",
        "veracity_label",
        "selection_score",
        "selection_scores",
        "selection_probability",
        "selection_probabilities",
        "selector_score",
        "selector_scores",
        "veracity_logits",
        "sample_logits",
        "logits",
        "probabilities",
        "prediction",
        "prediction_id",
        "top_k",
        "topk",
        "top_k_selection_units",
        "top_k_membership",
        "rank",
    }
)
_BOUNDARY_FALSE_FIELDS = (
    "selection_outputs_inspected",
    "veracity_labels_inspected",
    "formal_validation_accessed",
    "formal_test_accessed",
    "model_loaded",
    "checkpoint_loaded",
    "training_started",
)


class NeutralBuildError(ValueError):
    """Raised when source integrity or the scientific gate fails closed."""

    def __init__(
        self,
        message: str,
        *,
        report: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ExpectedCounts:
    total_cases: int
    total_examples: int
    train_cases: int
    dev_cases: int
    ocr_examples: int
    transcript_examples: int
    dataset_case_counts: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        numeric = (
            self.total_cases,
            self.total_examples,
            self.train_cases,
            self.dev_cases,
            self.ocr_examples,
            self.transcript_examples,
        )
        if any(type(value) is not int or value < 0 for value in numeric):
            raise NeutralBuildError("expected counts must be nonnegative integers")
        if self.total_cases != self.train_cases + self.dev_cases:
            raise NeutralBuildError("expected Train/Dev cases must sum to total cases")
        if self.total_examples != self.ocr_examples + self.transcript_examples:
            raise NeutralBuildError("expected modality examples must sum to total")
        names = [name for name, _ in self.dataset_case_counts]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise NeutralBuildError("expected dataset names must be unique and nonblank")
        if any(type(count) is not int or count < 0 for _, count in self.dataset_case_counts):
            raise NeutralBuildError("expected dataset counts must be nonnegative integers")
        if sum(count for _, count in self.dataset_case_counts) != self.total_cases:
            raise NeutralBuildError("expected dataset counts must sum to total cases")

    @property
    def dataset_counts(self) -> Mapping[str, int]:
        return dict(self.dataset_case_counts)


AUTHORITATIVE_COUNTS = ExpectedCounts(
    total_cases=1306,
    total_examples=5224,
    train_cases=1045,
    dev_cases=261,
    ocr_examples=2612,
    transcript_examples=2612,
    dataset_case_counts=(("GroundLie360", 570), ("TRUE-3MFact", 736)),
)


@dataclass(frozen=True)
class SourceArtifacts:
    source_dir: Path
    build_report: Mapping[str, Any]
    train_examples: Tuple[Mapping[str, Any], ...]
    dev_examples: Tuple[Mapping[str, Any], ...]
    split_assignments: Mapping[str, str]
    eligible_case_ids: Tuple[str, ...]
    heldout_case_ids: Tuple[str, ...]
    artifact_sha256: Mapping[str, str]
    protected_sha256: Mapping[str, str]

    @property
    def examples(self) -> Tuple[Mapping[str, Any], ...]:
        return self.train_examples + self.dev_examples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_safe_path(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    normalized = {
        re.sub(r"[^a-z0-9]+", "", part.casefold()) for part in resolved.parts
    }
    if normalized & _RESTRICTED_PATH_PARTS:
        raise NeutralBuildError(f"{field} must not reference Formal Validation/Test")
    return resolved


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NeutralBuildError(f"{field} is missing") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeutralBuildError(f"{field} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise NeutralBuildError(f"{field} must be a JSON object")
    return value


def _read_jsonl(path: Path, field: str) -> Tuple[Mapping[str, Any], ...]:
    records: List[Mapping[str, Any]] = []
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise NeutralBuildError(f"{field} is missing") from exc
    try:
        with stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise NeutralBuildError(f"{field} row {index} is malformed") from exc
                if not isinstance(record, dict):
                    raise NeutralBuildError(f"{field} row {index} must be an object")
                records.append(record)
    except UnicodeDecodeError as exc:
        raise NeutralBuildError(f"{field} is not valid UTF-8") from exc
    return tuple(records)


def _read_sidecar(path: Path, field: str) -> str:
    try:
        digest = path.read_text(encoding="utf-8").strip().casefold()
    except FileNotFoundError as exc:
        raise NeutralBuildError(f"{field} is missing") from exc
    if not _SHA256_RE.fullmatch(digest):
        raise NeutralBuildError(f"{field} must contain exactly one SHA-256")
    return digest


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NeutralBuildError(f"{field} must be a nonblank string")
    return value


def _string_list(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise NeutralBuildError(f"{field} must be a list")
    result = tuple(_nonblank(item, field) for item in value)
    if len(result) != len(set(result)):
        raise NeutralBuildError(f"{field} contains duplicate IDs")
    return result


def _assert_no_forbidden_scientific_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_SCIENTIFIC_KEYS:
                raise NeutralBuildError(f"forbidden scientific field in source: {key}")
            _assert_no_forbidden_scientific_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_scientific_keys(item)


def _expected_old_claim(modality: str, anchor_text: str) -> str:
    if modality == OCR:
        prefix = OCR_PREFIX
    elif modality == TRANSCRIPT:
        prefix = TRANSCRIPT_PREFIX
    else:
        raise NeutralBuildError("expected_modality must be OCR or TRANSCRIPT")
    return prefix + anchor_text + CLAIM_SUFFIX


def _validate_source_example(
    example: Mapping[str, Any],
    *,
    file_split: str,
    index: int,
) -> None:
    field = f"calibration {file_split} row {index}"
    _assert_no_forbidden_scientific_keys(example)
    _nonblank(example.get("calibration_example_id"), f"{field}.id")
    _nonblank(example.get("source_dataset"), f"{field}.source_dataset")
    _nonblank(example.get("source_case_id"), f"{field}.source_case_id")
    _nonblank(
        example.get("canonical_underlying_case_id"), f"{field}.underlying_case"
    )
    split = _nonblank(example.get("calibration_split"), f"{field}.split")
    if split != file_split:
        raise NeutralBuildError(f"{field} is stored under the wrong split")
    modality = _nonblank(example.get("expected_modality"), f"{field}.modality")
    anchor_id = _nonblank(example.get("anchor_unit_id"), f"{field}.anchor_id")
    anchor_text = _nonblank(example.get("anchor_text"), f"{field}.anchor_text")
    claim = _nonblank(example.get("claim"), f"{field}.claim")
    if claim != _expected_old_claim(modality, anchor_text):
        raise NeutralBuildError(f"{field} has an unknown or nonconforming old claim template")
    positive_ids = _string_list(example.get("positive_unit_ids"), f"{field}.positive")
    candidates = example.get("candidate_units")
    if not isinstance(candidates, list) or not candidates:
        raise NeutralBuildError(f"{field}.candidate_units must be a nonempty list")
    candidate_ids: List[str] = []
    target_positive_ids: List[str] = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise NeutralBuildError(f"{field}.candidate[{candidate_index}] must be an object")
        candidate_id = _nonblank(
            candidate.get("unit_id"), f"{field}.candidate[{candidate_index}].unit_id"
        )
        _nonblank(
            candidate.get("unit_type"), f"{field}.candidate[{candidate_index}].unit_type"
        )
        _nonblank(
            candidate.get("modality"), f"{field}.candidate[{candidate_index}].modality"
        )
        _nonblank(candidate.get("text"), f"{field}.candidate[{candidate_index}].text")
        target = candidate.get("relevance_target")
        if type(target) is not int or target not in {0, 1}:
            raise NeutralBuildError(
                f"{field}.candidate[{candidate_index}].relevance_target must be 0 or 1"
            )
        candidate_ids.append(candidate_id)
        if target == 1:
            target_positive_ids.append(candidate_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise NeutralBuildError(f"{field} contains duplicate candidate IDs")
    if tuple(target_positive_ids) != positive_ids:
        raise NeutralBuildError(f"{field} positive IDs disagree with relevance targets")
    if anchor_id not in positive_ids:
        raise NeutralBuildError(f"{field} anchor is not a positive candidate")
    exposed_count = example.get("model_exposed_candidate_count")
    if type(exposed_count) is not int or exposed_count != len(candidates):
        raise NeutralBuildError(f"{field} model exposure count is inconsistent")


def _verify_declared_hash(
    source_dir: Path,
    report_hashes: Mapping[str, Any],
    name: str,
) -> str:
    actual = sha256_file(source_dir / name)
    if report_hashes.get(name) != actual:
        raise NeutralBuildError(f"build report SHA mismatch for {name}")
    return actual


def load_source_artifacts(source_dir: Path) -> SourceArtifacts:
    source_dir = _resolve_safe_path(source_dir, "source calibration directory")
    if not source_dir.is_dir():
        raise NeutralBuildError("source calibration directory is missing")
    for name in _REQUIRED_SOURCE_FILES:
        if not (source_dir / name).is_file():
            raise NeutralBuildError(f"required source artifact is missing: {name}")
    build_report = _read_json(source_dir / "build_report.json", "source build report")
    if build_report.get("status") != "COMPLETED":
        raise NeutralBuildError("source calibration build is not completed")
    if build_report.get("implementation_revision") != SOURCE_IMPLEMENTATION_REVISION:
        raise NeutralBuildError("source calibration implementation revision is not v2")
    for field in _BOUNDARY_FALSE_FIELDS:
        if build_report.get(field) is not False:
            raise NeutralBuildError(f"source build report {field} must be false")
    report_hashes = build_report.get("artifact_sha256")
    if not isinstance(report_hashes, Mapping):
        raise NeutralBuildError("source build report artifact_sha256 is required")
    artifact_hashes: Dict[str, str] = {}
    for name in (
        "calibration_train.jsonl",
        "calibration_dev.jsonl",
        "calibration_split_manifest.json",
        "eligible_case_manifest.json",
    ):
        artifact_hashes[name] = _verify_declared_hash(source_dir, report_hashes, name)
    for stem in ("calibration_train", "calibration_dev"):
        sidecar = _read_sidecar(source_dir / f"{stem}.sha256", f"{stem} sidecar")
        if sidecar != artifact_hashes[f"{stem}.jsonl"]:
            raise NeutralBuildError(f"SHA sidecar mismatch for {stem}.jsonl")
    for sidecar_name in _OPTIONAL_SOURCE_SIDECARS:
        sidecar_path = source_dir / sidecar_name
        if not sidecar_path.exists():
            continue
        stem = sidecar_name[: -len(".sha256")]
        suffix = ".json"
        sidecar = _read_sidecar(sidecar_path, f"{stem} sidecar")
        if sidecar != artifact_hashes[f"{stem}{suffix}"]:
            raise NeutralBuildError(f"SHA sidecar mismatch for {stem}.json")
    split_manifest = _read_json(
        source_dir / "calibration_split_manifest.json", "calibration split manifest"
    )
    raw_assignments = split_manifest.get("assignments")
    if not isinstance(raw_assignments, list):
        raise NeutralBuildError("split manifest assignments must be a list")
    assignments: Dict[str, str] = {}
    assignment_datasets: Dict[str, str] = {}
    for index, item in enumerate(raw_assignments):
        if not isinstance(item, Mapping):
            raise NeutralBuildError(f"split assignment {index} must be an object")
        identity = _nonblank(
            item.get("canonical_underlying_case_id"), f"split assignment {index}.case"
        )
        dataset = _nonblank(item.get("source_dataset"), f"split assignment {index}.dataset")
        split = _nonblank(item.get("calibration_split"), f"split assignment {index}.split")
        if split not in {"train", "dev"} or identity in assignments:
            raise NeutralBuildError("split manifest contains invalid assignments")
        assignments[identity] = split
        assignment_datasets[identity] = dataset
    eligible_manifest = _read_json(
        source_dir / "eligible_case_manifest.json", "eligible case manifest"
    )
    raw_eligible = eligible_manifest.get("eligible_cases")
    if not isinstance(raw_eligible, list):
        raise NeutralBuildError("eligible case manifest is malformed")
    eligible_ids: List[str] = []
    for index, item in enumerate(raw_eligible):
        if not isinstance(item, Mapping):
            raise NeutralBuildError(f"eligible case {index} must be an object")
        identity = _nonblank(
            item.get("canonical_underlying_case_id"), f"eligible case {index}.case"
        )
        dataset = _nonblank(item.get("source_dataset"), f"eligible case {index}.dataset")
        if identity in eligible_ids or assignment_datasets.get(identity) != dataset:
            raise NeutralBuildError("eligible and split manifests are inconsistent")
        eligible_ids.append(identity)
    if set(eligible_ids) != set(assignments):
        raise NeutralBuildError("eligible and split manifest case identities differ")
    train = _read_jsonl(source_dir / "calibration_train.jsonl", "calibration Train")
    dev = _read_jsonl(source_dir / "calibration_dev.jsonl", "calibration Dev")
    seen_example_ids = set()
    train_case_ids = set()
    dev_case_ids = set()
    for split, examples, case_ids in (
        ("train", train, train_case_ids),
        ("dev", dev, dev_case_ids),
    ):
        for index, example in enumerate(examples):
            _validate_source_example(example, file_split=split, index=index)
            example_id = str(example["calibration_example_id"])
            if example_id in seen_example_ids:
                raise NeutralBuildError("duplicate calibration_example_id")
            seen_example_ids.add(example_id)
            canonical = str(example["canonical_underlying_case_id"])
            case_ids.add(canonical)
    overlap = train_case_ids & dev_case_ids
    if overlap:
        raise NeutralBuildError("source calibration has Train/Dev overlap")
    heldout_raw = build_report.get("heldout_case_ids")
    if not isinstance(heldout_raw, list):
        raise NeutralBuildError("source build report heldout_case_ids is required")
    heldout_ids = _string_list(heldout_raw, "heldout_case_ids")
    source_case_ids = train_case_ids | dev_case_ids
    if source_case_ids & set(heldout_ids):
        raise NeutralBuildError("source calibration has held-out overlap")
    for split, examples in (("train", train), ("dev", dev)):
        for example in examples:
            canonical = str(example["canonical_underlying_case_id"])
            if assignments.get(canonical) != split:
                raise NeutralBuildError("source example disagrees with split manifest")
            if assignment_datasets.get(canonical) != example.get("source_dataset"):
                raise NeutralBuildError("source example disagrees with dataset manifest")
    if source_case_ids != set(assignments):
        raise NeutralBuildError("source examples and manifests contain different cases")
    protected_names = list(_REQUIRED_SOURCE_FILES)
    protected_names.extend(
        name for name in _OPTIONAL_SOURCE_SIDECARS if (source_dir / name).is_file()
    )
    protected_hashes = {
        name: sha256_file(source_dir / name) for name in sorted(protected_names)
    }
    return SourceArtifacts(
        source_dir=source_dir,
        build_report=build_report,
        train_examples=train,
        dev_examples=dev,
        split_assignments=assignments,
        eligible_case_ids=tuple(eligible_ids),
        heldout_case_ids=heldout_ids,
        artifact_sha256=dict(sorted(artifact_hashes.items())),
        protected_sha256=protected_hashes,
    )


def neutralize_example(
    example: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    modality = _nonblank(example.get("expected_modality"), "expected_modality")
    anchor_text = _nonblank(example.get("anchor_text"), "anchor_text")
    old_claim = _nonblank(example.get("claim"), "claim")
    if old_claim != _expected_old_claim(modality, anchor_text):
        raise NeutralBuildError("unknown or nonconforming old claim template")
    new_claim = NEUTRAL_PREFIX + anchor_text + CLAIM_SUFFIX
    derived = dict(example)
    derived["claim"] = new_claim
    source_candidates = example["candidate_units"]
    derived_candidates = derived["candidate_units"]
    source_candidate_ids = tuple(item["unit_id"] for item in source_candidates)
    derived_candidate_ids = tuple(item["unit_id"] for item in derived_candidates)
    source_targets = tuple(
        (item["unit_id"], item["relevance_target"]) for item in source_candidates
    )
    derived_targets = tuple(
        (item["unit_id"], item["relevance_target"]) for item in derived_candidates
    )
    source_without_claim = {key: value for key, value in example.items() if key != "claim"}
    derived_without_claim = {key: value for key, value in derived.items() if key != "claim"}
    manifest = {
        "calibration_example_id": example["calibration_example_id"],
        "expected_modality": modality,
        "old_claim": old_claim,
        "new_claim": new_claim,
        "anchor_text": anchor_text,
        "anchor_text_unchanged": derived["anchor_text"] == example["anchor_text"],
        "candidate_ids_unchanged": set(source_candidate_ids) == set(derived_candidate_ids),
        "candidate_order_unchanged": source_candidate_ids == derived_candidate_ids,
        "candidate_content_unchanged": source_candidates == derived_candidates,
        "positive_ids_unchanged": tuple(example["positive_unit_ids"])
        == tuple(derived["positive_unit_ids"]),
        "relevance_targets_unchanged": source_targets == derived_targets,
        "underlying_case_unchanged": example["canonical_underlying_case_id"]
        == derived["canonical_underlying_case_id"],
        "split_unchanged": example["calibration_split"]
        == derived["calibration_split"],
        "all_non_claim_content_unchanged": source_without_claim == derived_without_claim,
    }
    return derived, manifest


def _neutral_prefix(claim: str, anchor_text: str) -> str:
    suffix = anchor_text + CLAIM_SUFFIX
    if not claim.endswith(suffix):
        raise NeutralBuildError("neutral claim does not preserve the exact anchor text")
    return claim[: -len(suffix)]


def _prefix_only_modality_accuracy(
    examples: Sequence[Mapping[str, Any]],
) -> Tuple[float, int]:
    prefix_modalities: Dict[str, set[str]] = defaultdict(set)
    prefixes: List[str] = []
    for example in examples:
        prefix = _neutral_prefix(str(example["claim"]), str(example["anchor_text"]))
        prefixes.append(prefix)
        prefix_modalities[prefix].add(str(example["expected_modality"]))
    correct = 0
    for prefix, example in zip(prefixes, examples):
        modalities = prefix_modalities[prefix]
        if len(modalities) == 1 and str(example["expected_modality"]) in modalities:
            correct += 1
    denominator = len(examples) or 1
    return correct / denominator, len(prefix_modalities)


def _case_statistics(
    examples: Sequence[Mapping[str, Any]],
) -> Tuple[set[str], set[str], Mapping[str, int], Mapping[str, Mapping[str, int]]]:
    case_records: Dict[str, Tuple[str, str, str]] = {}
    for example in examples:
        canonical = str(example["canonical_underlying_case_id"])
        record = (
            str(example["source_dataset"]),
            str(example["source_case_id"]),
            str(example["calibration_split"]),
        )
        if canonical in case_records and case_records[canonical] != record:
            raise NeutralBuildError("case metadata differs across calibration examples")
        case_records[canonical] = record
    train_cases = {
        identity for identity, (_, _, split) in case_records.items() if split == "train"
    }
    dev_cases = {
        identity for identity, (_, _, split) in case_records.items() if split == "dev"
    }
    dataset_counts = Counter(dataset for dataset, _, _ in case_records.values())
    split_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"train": 0, "dev": 0}
    )
    for dataset, _, split in case_records.values():
        split_counts[dataset][split] += 1
    return (
        train_cases,
        dev_cases,
        dict(sorted(dataset_counts.items())),
        {dataset: counts for dataset, counts in sorted(split_counts.items())},
    )


def _require_report_matches_source(
    report: Mapping[str, Any],
    *,
    train_count: int,
    dev_count: int,
    train_cases: set[str],
    dev_cases: set[str],
    dataset_counts: Mapping[str, int],
    dataset_split_counts: Mapping[str, Mapping[str, int]],
    modality_counts: Mapping[str, int],
) -> None:
    comparisons = {
        "eligible_case_count": len(train_cases | dev_cases),
        "calibration_train_case_count": len(train_cases),
        "calibration_dev_case_count": len(dev_cases),
        "calibration_train_example_count": train_count,
        "calibration_dev_example_count": dev_count,
        "ocr_example_count": modality_counts.get(OCR, 0),
        "transcript_example_count": modality_counts.get(TRANSCRIPT, 0),
    }
    for field, expected in comparisons.items():
        if report.get(field) != expected:
            raise NeutralBuildError(f"source build report {field} disagrees with artifacts")
    if report.get("dataset_case_counts") != dataset_counts:
        raise NeutralBuildError("source build report dataset_case_counts disagrees")
    if report.get("dataset_split_counts") != dataset_split_counts:
        raise NeutralBuildError("source build report dataset_split_counts disagrees")


def _dataset_card(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# G1-RelevanceSelector Modality-Neutral Calibration v1",
            "",
            "## Purpose",
            "",
            "A deterministic claim-only neutral revision of the closed Step 2.6R-1A v2 calibration artifact.",
            "",
            "## Transformation",
            "",
            f'- Both OCR and Transcript examples use: `{NEUTRAL_PREFIX}<ANCHOR_TEXT>{CLAIM_SUFFIX}`',
            "- Candidate units, modality fields, target labels, provenance, and Train/Dev assignments are unchanged.",
            "",
            "## Boundaries",
            "",
            "- No selector or veracity outputs were inspected.",
            "- No model or checkpoint was loaded.",
            "- Formal Validation and Formal Test were not accessed.",
            "- No training was performed.",
            f"- Source examples: {report['source_example_count']}.",
            f"- Neutral examples: {report['neutral_example_count']}.",
            f"- Prefix-only modality accuracy: {report['claim_only_template_modality_accuracy_after_neutralization']}.",
            "",
        ]
    )


def _write_bytes(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return _sha256_bytes(content)


def build_neutral_calibration(
    *,
    source_dir: Path,
    output_dir: Path,
    expected_counts: ExpectedCounts = AUTHORITATIVE_COUNTS,
) -> Mapping[str, Any]:
    output_dir = _resolve_safe_path(output_dir, "neutral output directory")
    artifacts = load_source_artifacts(source_dir)
    source_resolved = artifacts.source_dir
    if (
        output_dir == source_resolved
        or output_dir in source_resolved.parents
        or source_resolved in output_dir.parents
    ):
        raise NeutralBuildError("neutral output must be isolated from source artifacts")
    if output_dir.exists() and not output_dir.is_dir():
        raise NeutralBuildError("neutral output path must be a directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise NeutralBuildError("neutral output directory must be absent or empty")
    source_examples = artifacts.examples
    source_train_cases, source_dev_cases, dataset_counts, dataset_split_counts = (
        _case_statistics(source_examples)
    )
    modality_counts = Counter(str(item["expected_modality"]) for item in source_examples)
    _require_report_matches_source(
        artifacts.build_report,
        train_count=len(artifacts.train_examples),
        dev_count=len(artifacts.dev_examples),
        train_cases=source_train_cases,
        dev_cases=source_dev_cases,
        dataset_counts=dataset_counts,
        dataset_split_counts=dataset_split_counts,
        modality_counts=modality_counts,
    )
    neutral_train: List[Mapping[str, Any]] = []
    neutral_dev: List[Mapping[str, Any]] = []
    manifest_rows: List[Mapping[str, Any]] = []
    for source, target in (
        (artifacts.train_examples, neutral_train),
        (artifacts.dev_examples, neutral_dev),
    ):
        for example in source:
            neutral, manifest = neutralize_example(example)
            target.append(neutral)
            manifest_rows.append(manifest)
    neutral_examples = tuple(neutral_train + neutral_dev)
    neutral_train_cases, neutral_dev_cases, neutral_dataset_counts, neutral_split_counts = (
        _case_statistics(neutral_examples)
    )
    prefix_accuracy, prefix_count = _prefix_only_modality_accuracy(neutral_examples)
    failure_fields = {
        "anchor_text_invariance_failures": "anchor_text_unchanged",
        "candidate_id_invariance_failures": "candidate_ids_unchanged",
        "candidate_order_invariance_failures": "candidate_order_unchanged",
        "candidate_content_invariance_failures": "candidate_content_unchanged",
        "positive_id_invariance_failures": "positive_ids_unchanged",
        "relevance_target_invariance_failures": "relevance_targets_unchanged",
        "underlying_case_invariance_failures": "underlying_case_unchanged",
        "split_invariance_failures": "split_unchanged",
        "non_claim_content_invariance_failures": "all_non_claim_content_unchanged",
    }
    failures = {
        report_field: sum(not bool(row[manifest_field]) for row in manifest_rows)
        for report_field, manifest_field in failure_fields.items()
    }
    source_cases = source_train_cases | source_dev_cases
    neutral_cases = neutral_train_cases | neutral_dev_cases
    heldout_overlap_ids = sorted(neutral_cases & set(artifacts.heldout_case_ids))
    train_dev_overlap_ids = sorted(neutral_train_cases & neutral_dev_cases)
    claim_changed_count = sum(
        source["claim"] != neutral["claim"]
        for source, neutral in zip(source_examples, neutral_examples)
    )
    claim_unchanged_count = len(source_examples) - claim_changed_count
    report: Dict[str, Any] = {
        "status": "PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "source_calibration_train_sha256": artifacts.artifact_sha256[
            "calibration_train.jsonl"
        ],
        "source_calibration_dev_sha256": artifacts.artifact_sha256[
            "calibration_dev.jsonl"
        ],
        "source_artifact_sha256": dict(artifacts.artifact_sha256),
        "source_protected_file_sha256": dict(artifacts.protected_sha256),
        "source_heldout_case_ids": list(artifacts.heldout_case_ids),
        "source_example_count": len(source_examples),
        "neutral_example_count": len(neutral_examples),
        "source_case_count": len(source_cases),
        "neutral_case_count": len(neutral_cases),
        "train_case_count": len(neutral_train_cases),
        "dev_case_count": len(neutral_dev_cases),
        "dataset_case_counts": neutral_dataset_counts,
        "dataset_split_counts": neutral_split_counts,
        "ocr_example_count": modality_counts.get(OCR, 0),
        "transcript_example_count": modality_counts.get(TRANSCRIPT, 0),
        "claim_changed_count": claim_changed_count,
        "claim_unchanged_count": claim_unchanged_count,
        "neutral_template": NEUTRAL_PREFIX + "<ANCHOR_TEXT>" + CLAIM_SUFFIX,
        "unique_neutral_template_prefix_count": prefix_count,
        "claim_only_template_modality_metric": (
            "PREFIX_UNIQUE_MODALITY_LOOKUP_AMBIGUOUS_AS_UNKNOWN"
        ),
        "claim_only_template_modality_accuracy_after_neutralization": prefix_accuracy,
        **failures,
        "heldout_overlap": len(heldout_overlap_ids),
        "heldout_overlap_case_ids": heldout_overlap_ids,
        "train_dev_overlap": len(train_dev_overlap_ids),
        "train_dev_overlap_case_ids": train_dev_overlap_ids,
        "selection_outputs_inspected": False,
        "veracity_labels_inspected": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "training_started": False,
        "production_or_model_code_changed": False,
    }
    gate_failures: List[str] = []
    expected_comparisons = {
        "source_example_count": expected_counts.total_examples,
        "neutral_example_count": expected_counts.total_examples,
        "source_case_count": expected_counts.total_cases,
        "neutral_case_count": expected_counts.total_cases,
        "train_case_count": expected_counts.train_cases,
        "dev_case_count": expected_counts.dev_cases,
        "ocr_example_count": expected_counts.ocr_examples,
        "transcript_example_count": expected_counts.transcript_examples,
        "claim_changed_count": expected_counts.total_examples,
        "claim_unchanged_count": 0,
        "unique_neutral_template_prefix_count": 1,
        "heldout_overlap": 0,
        "train_dev_overlap": 0,
    }
    for field, expected in expected_comparisons.items():
        if report[field] != expected:
            gate_failures.append(field)
    if neutral_dataset_counts != expected_counts.dataset_counts:
        gate_failures.append("dataset_case_counts")
    for field in failures:
        if report[field] != 0:
            gate_failures.append(field)
    if prefix_accuracy >= 0.99:
        gate_failures.append("claim_only_template_modality_accuracy_after_neutralization")
    if source_cases != neutral_cases or source_train_cases != neutral_train_cases:
        gate_failures.append("case_identity_or_train_assignment")
    if source_dev_cases != neutral_dev_cases:
        gate_failures.append("dev_assignment")
    if gate_failures:
        report["status"] = "FAIL"
        raise NeutralBuildError(
            "neutral scientific gate failed: " + ", ".join(sorted(set(gate_failures))),
            report=report,
        )
    manifest_rows.sort(key=lambda item: str(item["calibration_example_id"]))
    manifest_payload = {
        "schema_version": 1,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "neutral_template": report["neutral_template"],
        "example_count": len(manifest_rows),
        "examples": manifest_rows,
    }
    neutral_train_bytes = _jsonl_bytes(neutral_train)
    neutral_dev_bytes = _jsonl_bytes(neutral_dev)
    manifest_bytes = _json_bytes(manifest_payload)
    report["neutral_train_sha256"] = _sha256_bytes(neutral_train_bytes)
    report["neutral_dev_sha256"] = _sha256_bytes(neutral_dev_bytes)
    manifest_sha = _sha256_bytes(manifest_bytes)
    dataset_card_bytes = _dataset_card(report).encode("utf-8")
    report["artifact_sha256"] = {
        "neutral_calibration_train.jsonl": report["neutral_train_sha256"],
        "neutral_calibration_dev.jsonl": report["neutral_dev_sha256"],
        "neutral_revision_manifest.json": manifest_sha,
        "dataset_card.md": _sha256_bytes(dataset_card_bytes),
    }
    before_write_hashes = {
        name: sha256_file(source_resolved / name)
        for name in artifacts.protected_sha256
    }
    if before_write_hashes != artifacts.protected_sha256:
        raise NeutralBuildError("source artifacts changed during neutral revision")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_bytes(output_dir / "neutral_calibration_train.jsonl", neutral_train_bytes)
    (output_dir / "neutral_calibration_train.sha256").write_text(
        report["neutral_train_sha256"] + "\n", encoding="utf-8"
    )
    _write_bytes(output_dir / "neutral_calibration_dev.jsonl", neutral_dev_bytes)
    (output_dir / "neutral_calibration_dev.sha256").write_text(
        report["neutral_dev_sha256"] + "\n", encoding="utf-8"
    )
    _write_bytes(output_dir / "neutral_revision_manifest.json", manifest_bytes)
    (output_dir / "neutral_revision_manifest.sha256").write_text(
        manifest_sha + "\n", encoding="utf-8"
    )
    _write_bytes(output_dir / "dataset_card.md", dataset_card_bytes)
    _write_bytes(output_dir / "neutral_build_report.json", _json_bytes(report))
    after_write_hashes = {
        name: sha256_file(source_resolved / name)
        for name in artifacts.protected_sha256
    }
    if after_write_hashes != artifacts.protected_sha256:
        raise NeutralBuildError("source artifacts changed while outputs were written")
    return report
