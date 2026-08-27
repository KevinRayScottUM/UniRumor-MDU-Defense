"""Audit deterministic claim-template and neural-input shortcut exposure.

This package is deliberately read-only with respect to the authoritative
calibration artifacts.  It never imports or executes the Frozen G1 model,
loads a checkpoint/tokenizer, or consults selector/veracity outputs.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


IMPLEMENTATION_REVISION = "step2.6r-1c-v1"
CALIBRATION_REVISION = "step2.6r-1a-v2"
OCR = "OCR"
TRANSCRIPT = "TRANSCRIPT"
UNKNOWN = "UNKNOWN"
OCR_PREFIX = 'The on-screen text reads "'
TRANSCRIPT_PREFIX = 'The speaker says "'
NEUTRAL_PREFIX = 'The relevant content states "'
CLAIM_SUFFIX = '".'
UNIT_MODALITY_ENCODINGS = {"EXPLICIT", "IMPLICIT_ONLY", UNKNOWN}
SHORTCUT_RISKS = {
    "HIGH_TEMPLATE_MODALITY_SHORTCUT_RISK",
    "MODERATE_TEMPLATE_MODALITY_SHORTCUT_RISK",
    "LOW_TEMPLATE_MODALITY_SHORTCUT_RISK",
    "INCONCLUSIVE",
}
TRAINING_RECOMMENDATIONS = {
    "ORIGINAL_TEMPLATE_TRAINING_ACCEPTABLE",
    "REQUIRE_TEMPLATE_NEUTRAL_CALIBRATION_BEFORE_TRAINING",
    "REQUIRE_FURTHER_ENCODING_AUDIT",
}
EXPECTED_HELDOUT_IDS = frozenset(
    {
        "GroundLie360:13025004",
        "TRUE-3MFact:10145403",
        "TRUE-3MFact:10258205",
        "TRUE-3MFact:10372904",
        "TRUE-3MFact:10455808",
        "TRUE-3MFact:10865013",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORMAL_PATH_PARTS = {"validation", "test", "formalvalidation", "formaltest"}
_REQUIRED_CALIBRATION_FILES = (
    "build_report.json",
    "calibration_train.jsonl",
    "calibration_train.sha256",
    "calibration_dev.jsonl",
    "calibration_dev.sha256",
    "calibration_split_manifest.json",
    "eligible_case_manifest.json",
)
_SOURCE_RELATIVE_PATHS = (
    Path("MDU/scripts/clip12_phase3_common/clip12p3_model.py"),
    Path("MDU/scripts/clip12_phase3a_final_fit/clip12p3a_final_fit.py"),
    Path("MDU/scripts/clip12_phase4a_inference_handoff/clip12p4a_engine.py"),
    Path("MDU/configs/clip12_phase4a_frozen_g1_inference_handoff.json"),
)


class AuditInputError(ValueError):
    """Raised when the audit cannot preserve its registered boundaries."""


@dataclass(frozen=True)
class ExpectedCounts:
    total: int
    ocr: int
    transcript: int

    def __post_init__(self) -> None:
        values = (self.total, self.ocr, self.transcript)
        if any(type(value) is not int or value < 0 for value in values):
            raise AuditInputError("expected counts must be nonnegative integers")
        if self.total != self.ocr + self.transcript:
            raise AuditInputError("expected modality counts must sum to total")


AUTHORITATIVE_COUNTS = ExpectedCounts(total=5224, ocr=2612, transcript=2612)


@dataclass(frozen=True)
class TemplateParse:
    modality: str
    anchor_text: str
    prefix: str


@dataclass(frozen=True)
class CalibrationArtifacts:
    calibration_dir: Path
    build_report: Mapping[str, Any]
    train_examples: Tuple[Mapping[str, Any], ...]
    dev_examples: Tuple[Mapping[str, Any], ...]
    split_assignments: Mapping[str, str]
    file_sha256: Mapping[str, str]

    @property
    def examples(self) -> Tuple[Mapping[str, Any], ...]:
        return self.train_examples + self.dev_examples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).strip()


def _reject_formal_path(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    normalized_parts = {
        re.sub(r"[^a-z0-9]+", "", part.casefold()) for part in resolved.parts
    }
    if normalized_parts & _FORMAL_PATH_PARTS:
        raise AuditInputError(f"{field} must not reference Formal Validation/Test")
    return resolved


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditInputError(f"{field} is missing") from exc
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"{field} is malformed JSON") from exc
    if not isinstance(payload, dict):
        raise AuditInputError(f"{field} must be a JSON object")
    return payload


def _read_jsonl(path: Path, field: str) -> Tuple[Mapping[str, Any], ...]:
    records: List[Mapping[str, Any]] = []
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AuditInputError(f"{field} is missing") from exc
    with stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditInputError(f"{field} row {index} is malformed") from exc
            if not isinstance(record, dict):
                raise AuditInputError(f"{field} row {index} must be an object")
            records.append(record)
    return tuple(records)


def _read_sidecar(path: Path, field: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip().casefold()
    except FileNotFoundError as exc:
        raise AuditInputError(f"{field} is missing") from exc
    if not _SHA256_RE.fullmatch(value):
        raise AuditInputError(f"{field} must contain one SHA-256 digest")
    return value


def _require_false(mapping: Mapping[str, Any], field: str) -> None:
    if mapping.get(field) is not False:
        raise AuditInputError(f"calibration report {field} must be false")


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditInputError(f"{field} must be a non-blank string")
    return value.strip()


def _string_list(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise AuditInputError(f"{field} must be a list")
    result = tuple(_nonblank(item, field) for item in value)
    if len(result) != len(set(result)):
        raise AuditInputError(f"{field} contains duplicate IDs")
    return result


def _validate_example(
    example: Mapping[str, Any],
    split_assignments: Mapping[str, str],
    field: str,
) -> None:
    example_id = _nonblank(example.get("calibration_example_id"), f"{field}.id")
    modality = _nonblank(example.get("expected_modality"), f"{field}.modality")
    if modality not in {OCR, TRANSCRIPT}:
        raise AuditInputError(f"{field} has unsupported expected_modality")
    _nonblank(example.get("claim"), f"{field}.claim")
    anchor_id = _nonblank(example.get("anchor_unit_id"), f"{field}.anchor_unit_id")
    _nonblank(example.get("anchor_text"), f"{field}.anchor_text")
    canonical = _nonblank(
        example.get("canonical_underlying_case_id"), f"{field}.case"
    )
    split = _nonblank(example.get("calibration_split"), f"{field}.split")
    if split not in {"train", "dev"}:
        raise AuditInputError(f"{field} has invalid calibration_split")
    if split_assignments.get(canonical) != split:
        raise AuditInputError(f"{field} disagrees with calibration split manifest")
    if canonical in EXPECTED_HELDOUT_IDS:
        raise AuditInputError(f"held-out identity entered calibration: {canonical}")
    positive_ids = _string_list(example.get("positive_unit_ids"), f"{field}.positive")
    candidates = example.get("candidate_units")
    if not isinstance(candidates, list) or not candidates:
        raise AuditInputError(f"{field}.candidate_units must be non-empty")
    candidate_ids: List[str] = []
    target_positive_ids: List[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise AuditInputError(f"{field}.candidate[{index}] must be an object")
        unit_id = _nonblank(
            candidate.get("unit_id"), f"{field}.candidate[{index}].unit_id"
        )
        target = candidate.get("relevance_target")
        if type(target) is not int or target not in {0, 1}:
            raise AuditInputError(f"{field}.candidate[{index}] target must be 0 or 1")
        candidate_ids.append(unit_id)
        if target == 1:
            target_positive_ids.append(unit_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AuditInputError(f"{field} contains duplicate candidate IDs")
    if tuple(target_positive_ids) != positive_ids:
        raise AuditInputError(f"{field} positive IDs disagree with relevance targets")
    if anchor_id not in positive_ids:
        raise AuditInputError(f"{field} anchor is not a positive candidate")
    if not example_id:
        raise AuditInputError(f"{field} ID is missing")


def load_calibration_artifacts(calibration_dir: Path) -> CalibrationArtifacts:
    calibration_dir = _reject_formal_path(calibration_dir, "calibration directory")
    if not calibration_dir.is_dir():
        raise AuditInputError("calibration directory is missing")
    for name in _REQUIRED_CALIBRATION_FILES:
        if not (calibration_dir / name).is_file():
            raise AuditInputError(f"required calibration artifact is missing: {name}")
    report = _read_json(calibration_dir / "build_report.json", "build report")
    if report.get("implementation_revision") != CALIBRATION_REVISION:
        raise AuditInputError("calibration implementation revision is not v2")
    for field in (
        "selection_outputs_inspected",
        "veracity_labels_inspected",
        "formal_validation_accessed",
        "formal_test_accessed",
        "model_loaded",
        "checkpoint_loaded",
        "training_started",
    ):
        _require_false(report, field)
    report_hashes = report.get("artifact_sha256")
    if not isinstance(report_hashes, Mapping):
        raise AuditInputError("build report artifact_sha256 is required")
    artifact_names = (
        "calibration_train.jsonl",
        "calibration_dev.jsonl",
        "calibration_split_manifest.json",
        "eligible_case_manifest.json",
    )
    hashes: Dict[str, str] = {}
    for name in artifact_names:
        actual = sha256_file(calibration_dir / name)
        declared = report_hashes.get(name)
        if declared != actual:
            raise AuditInputError(f"build report SHA mismatch for {name}")
        hashes[name] = actual
    for stem in ("calibration_train", "calibration_dev"):
        declared = _read_sidecar(
            calibration_dir / f"{stem}.sha256", f"{stem} SHA sidecar"
        )
        if declared != hashes[f"{stem}.jsonl"]:
            raise AuditInputError(f"SHA sidecar mismatch for {stem}.jsonl")
    split_manifest = _read_json(
        calibration_dir / "calibration_split_manifest.json", "split manifest"
    )
    assignments_raw = split_manifest.get("assignments")
    if not isinstance(assignments_raw, list):
        raise AuditInputError("split manifest assignments must be a list")
    assignments: Dict[str, str] = {}
    for index, item in enumerate(assignments_raw):
        if not isinstance(item, Mapping):
            raise AuditInputError(f"split assignment {index} must be an object")
        identity = _nonblank(
            item.get("canonical_underlying_case_id"), f"split assignment {index}"
        )
        split = _nonblank(item.get("calibration_split"), f"split assignment {index}")
        if split not in {"train", "dev"} or identity in assignments:
            raise AuditInputError("split manifest contains invalid assignments")
        assignments[identity] = split
    eligible_manifest = _read_json(
        calibration_dir / "eligible_case_manifest.json", "eligible case manifest"
    )
    eligible = eligible_manifest.get("eligible_cases")
    if not isinstance(eligible, list):
        raise AuditInputError("eligible case manifest is malformed")
    eligible_ids = set()
    for index, item in enumerate(eligible):
        if not isinstance(item, Mapping):
            raise AuditInputError(f"eligible case {index} must be an object")
        identity = _nonblank(
            item.get("canonical_underlying_case_id"), f"eligible case {index}"
        )
        if identity in eligible_ids:
            raise AuditInputError("eligible case manifest contains duplicate identities")
        eligible_ids.add(identity)
    if eligible_ids != set(assignments):
        raise AuditInputError("eligible and split manifest case identities differ")
    train = _read_jsonl(calibration_dir / "calibration_train.jsonl", "calibration Train")
    dev = _read_jsonl(calibration_dir / "calibration_dev.jsonl", "calibration Dev")
    seen_ids = set()
    for split_name, examples in (("train", train), ("dev", dev)):
        for index, example in enumerate(examples):
            _validate_example(example, assignments, f"{split_name}[{index}]")
            example_id = str(example["calibration_example_id"])
            if example_id in seen_ids:
                raise AuditInputError("duplicate calibration_example_id")
            seen_ids.add(example_id)
            if example["calibration_split"] != split_name:
                raise AuditInputError("calibration JSONL is stored under the wrong split")
    return CalibrationArtifacts(
        calibration_dir=calibration_dir,
        build_report=report,
        train_examples=train,
        dev_examples=dev,
        split_assignments=assignments,
        file_sha256=hashes,
    )


def parse_original_template(claim: Any) -> Optional[TemplateParse]:
    if not isinstance(claim, str) or not claim.endswith(CLAIM_SUFFIX):
        return None
    for modality, prefix in ((OCR, OCR_PREFIX), (TRANSCRIPT, TRANSCRIPT_PREFIX)):
        if claim.startswith(prefix):
            anchor = claim[len(prefix) : -len(CLAIM_SUFFIX)]
            if anchor:
                return TemplateParse(modality=modality, anchor_text=anchor, prefix=prefix)
    return None


def predict_claim_only_modality(claim: Any) -> str:
    parsed = parse_original_template(claim)
    return parsed.modality if parsed is not None else UNKNOWN


def _observed_prefix_family(claim: str) -> str:
    parsed = parse_original_template(claim)
    if parsed is not None:
        return parsed.prefix
    quote_index = claim.find('"')
    return claim if quote_index < 0 else claim[: quote_index + 1]


def _invariant_snapshot(example: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = example["candidate_units"]
    return {
        "canonical_underlying_case_id": example["canonical_underlying_case_id"],
        "calibration_split": example["calibration_split"],
        "anchor_unit_id": example["anchor_unit_id"],
        "positive_unit_ids": tuple(example["positive_unit_ids"]),
        "candidate_unit_ids": tuple(candidate["unit_id"] for candidate in candidates),
        "relevance_targets": tuple(
            (candidate["unit_id"], candidate["relevance_target"])
            for candidate in candidates
        ),
    }


def _claim_for(prefix: str, anchor_text: str) -> str:
    return prefix + anchor_text + CLAIM_SUFFIX


def build_template_analysis(
    examples: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], Tuple[Mapping[str, Any], ...]]:
    total = len(examples)
    modality_counts = {OCR: 0, TRANSCRIPT: 0}
    prefix_families = {OCR: set(), TRANSCRIPT: set()}
    conforming = 0
    correct_claim_only = 0
    neutral_constructable = 0
    swapped_constructable = 0
    invariance_failures = 0
    manifest: List[Mapping[str, Any]] = []
    for index, example in enumerate(examples):
        modality = str(example["expected_modality"])
        claim = str(example["claim"])
        anchor_text = str(example["anchor_text"])
        modality_counts[modality] += 1
        prefix_families[modality].add(_observed_prefix_family(claim))
        parsed = parse_original_template(claim)
        expected_prefix = OCR_PREFIX if modality == OCR else TRANSCRIPT_PREFIX
        exact_claim = _claim_for(expected_prefix, anchor_text)
        exact_conformity = claim == exact_claim
        lexical_integrity = (
            parsed is not None
            and _normalize_text(parsed.anchor_text) == _normalize_text(anchor_text)
        )
        if exact_conformity and lexical_integrity:
            conforming += 1
        if predict_claim_only_modality(claim) == modality:
            correct_claim_only += 1
        neutral_claim = _claim_for(NEUTRAL_PREFIX, anchor_text)
        swapped_prefix = TRANSCRIPT_PREFIX if modality == OCR else OCR_PREFIX
        swapped_claim = _claim_for(swapped_prefix, anchor_text)
        original_snapshot = _invariant_snapshot(example)
        neutral_copy = dict(example)
        neutral_copy["claim"] = neutral_claim
        swapped_copy = dict(example)
        swapped_copy["claim"] = swapped_claim
        neutral_snapshot = _invariant_snapshot(neutral_copy)
        swapped_snapshot = _invariant_snapshot(swapped_copy)
        invariant_flags = {
            f"{field}_unchanged": (
                original_snapshot[field]
                == neutral_snapshot[field]
                == swapped_snapshot[field]
            )
            for field in original_snapshot
        }
        invariant = all(invariant_flags.values())
        anchor_preserved = (
            _normalize_text(anchor_text)
            == _normalize_text(neutral_claim[len(NEUTRAL_PREFIX) : -len(CLAIM_SUFFIX)])
            == _normalize_text(swapped_claim[len(swapped_prefix) : -len(CLAIM_SUFFIX)])
        )
        if invariant and anchor_preserved:
            neutral_constructable += 1
            swapped_constructable += 1
        else:
            invariance_failures += 1
        manifest.append(
            {
                "calibration_example_id": example["calibration_example_id"],
                "expected_modality": modality,
                "original_claim": claim,
                "neutral_claim": neutral_claim,
                "swapped_template_claim": swapped_claim,
                "anchor_text_unchanged": anchor_preserved,
                **invariant_flags,
                "targets_unchanged": invariant_flags["relevance_targets_unchanged"],
            }
        )
    manifest.sort(key=lambda item: str(item["calibration_example_id"]))
    denominator = total or 1
    report = {
        "total_example_count": total,
        "ocr_example_count": modality_counts[OCR],
        "transcript_example_count": modality_counts[TRANSCRIPT],
        "unique_ocr_template_prefix_count": len(prefix_families[OCR]),
        "unique_transcript_template_prefix_count": len(prefix_families[TRANSCRIPT]),
        "original_template_conformity_rate": conforming / denominator,
        "claim_only_template_modality_accuracy": correct_claim_only / denominator,
        "claim_only_correct_count": correct_claim_only,
        "neutral_control_template": _claim_for(NEUTRAL_PREFIX, "<TEXT>"),
        "swapped_control_definition": {
            OCR: _claim_for(TRANSCRIPT_PREFIX, "<OCR_ANCHOR_TEXT>"),
            TRANSCRIPT: _claim_for(OCR_PREFIX, "<TRANSCRIPT_ANCHOR_TEXT>"),
        },
        "neutral_control_constructable_count": neutral_constructable,
        "swapped_control_constructable_count": swapped_constructable,
        "control_target_invariance_failures": invariance_failures,
    }
    return report, tuple(manifest)


def classify_shortcut_risk(
    claim_only_template_modality_accuracy: float,
    unit_modality_encoding: str,
    *,
    encoding_contract_verified: bool = True,
) -> str:
    if unit_modality_encoding not in UNIT_MODALITY_ENCODINGS:
        raise AuditInputError("unsupported unit modality encoding classification")
    if not 0.0 <= claim_only_template_modality_accuracy <= 1.0:
        raise AuditInputError("claim-only modality accuracy must be between zero and one")
    if not encoding_contract_verified or unit_modality_encoding == UNKNOWN:
        return "INCONCLUSIVE"
    deterministic = claim_only_template_modality_accuracy >= 0.99
    if deterministic and unit_modality_encoding == "EXPLICIT":
        return "HIGH_TEMPLATE_MODALITY_SHORTCUT_RISK"
    if deterministic and unit_modality_encoding == "IMPLICIT_ONLY":
        return "MODERATE_TEMPLATE_MODALITY_SHORTCUT_RISK"
    if not deterministic and unit_modality_encoding == "IMPLICIT_ONLY":
        return "LOW_TEMPLATE_MODALITY_SHORTCUT_RISK"
    return "INCONCLUSIVE"


def recommend_training_action(shortcut_risk: str) -> str:
    if shortcut_risk not in SHORTCUT_RISKS:
        raise AuditInputError("unsupported shortcut risk classification")
    if shortcut_risk in {
        "HIGH_TEMPLATE_MODALITY_SHORTCUT_RISK",
        "MODERATE_TEMPLATE_MODALITY_SHORTCUT_RISK",
    }:
        return "REQUIRE_TEMPLATE_NEUTRAL_CALIBRATION_BEFORE_TRAINING"
    if shortcut_risk == "LOW_TEMPLATE_MODALITY_SHORTCUT_RISK":
        return "ORIGINAL_TEMPLATE_TRAINING_ACCEPTABLE"
    return "REQUIRE_FURTHER_ENCODING_AUDIT"


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _contains_role(source: str, role: str) -> bool:
    compact = source.casefold().replace(" ", "")
    if role == "claim":
        return "claim" in compact
    if role == "unit_text":
        return any(
            marker in compact
            for marker in (
                "unit_text",
                "unittext",
                "unit.text",
                "unit['text']",
                'unit["text"]',
                "candidate['text']",
                'candidate["text"]',
            )
        )
    return role.casefold() in compact


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


class _SourceAnalyzer(ast.NodeVisitor):
    def __init__(self, source: str, relative_path: str) -> None:
        self.source = source
        self.relative_path = relative_path
        self.scope: List[str] = []
        self.assignments: List[Dict[str, ast.AST]] = []
        self.tokenizer_calls: List[Mapping[str, Any]] = []
        self.tokenizer_preparation: List[Mapping[str, Any]] = []
        self.model_calls: List[Mapping[str, Any]] = []
        self.class_names: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_names.append(node.name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.assignments.append({})
        self.generic_visit(node)
        self.assignments.pop()
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.assignments:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignments[-1][target.id] = node.value
        self.generic_visit(node)

    def _expanded(self, node: ast.AST, depth: int = 0) -> str:
        if depth < 4 and isinstance(node, ast.Name) and self.assignments:
            assigned = self.assignments[-1].get(node.id)
            if assigned is not None:
                return f"{node.id}=({self._expanded(assigned, depth + 1)})"
        return ast.unparse(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        lowered = name.casefold()
        record = {
            "file": self.relative_path,
            "scope": ".".join(self.scope) or "<module>",
            "line_start": node.lineno,
            "line_end": getattr(node, "end_lineno", node.lineno),
            "call": ast.get_source_segment(self.source, node) or ast.unparse(node),
            "call_name": name,
            "positional_arguments": [self._expanded(item) for item in node.args],
            "keyword_arguments": {
                keyword.arg or "**": _literal_value(keyword.value)
                for keyword in node.keywords
            },
            "keyword_argument_expressions": {
                keyword.arg or "**": self._expanded(keyword.value)
                for keyword in node.keywords
            },
        }
        if lowered.endswith("tokenizer.from_pretrained") or lowered.endswith(
            "autotokenizer.from_pretrained"
        ):
            self.tokenizer_preparation.append(record)
        elif (
            lowered.endswith("tokenizer")
            or lowered.endswith("encode_plus")
            or lowered.endswith("batch_encode_plus")
        ):
            self.tokenizer_calls.append(record)
        if "mduselectorverifier" in lowered:
            self.model_calls.append(record)
        self.generic_visit(node)


def _recursive_config_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _recursive_config_value(nested, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for nested in value:
            found = _recursive_config_value(nested, key)
            if found is not None:
                return found
    return None


def inspect_encoding_contract(project_root: Path) -> Mapping[str, Any]:
    project_root = _reject_formal_path(project_root, "project root")
    if not project_root.is_dir():
        raise AuditInputError("project root is missing")
    source_hashes: Dict[str, str] = {}
    analyzers: List[_SourceAnalyzer] = []
    config: Mapping[str, Any] = {}
    for relative in _SOURCE_RELATIVE_PATHS:
        path = project_root / relative
        if not path.is_file():
            raise AuditInputError(f"authoritative encoding source is missing: {relative}")
        relative_text = relative.as_posix()
        source_hashes[relative_text] = sha256_file(path)
        if path.suffix == ".json":
            config = _read_json(path, "Phase4A configuration")
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=relative_text)
        except SyntaxError as exc:
            raise AuditInputError(f"cannot parse authoritative source: {relative}") from exc
        analyzer = _SourceAnalyzer(source, relative_text)
        analyzer.visit(tree)
        analyzers.append(analyzer)
    tokenizer_calls = [item for analyzer in analyzers for item in analyzer.tokenizer_calls]
    pair_candidates: List[Mapping[str, Any]] = []
    for call in tokenizer_calls:
        args = call["positional_arguments"]
        keyword_expressions = call["keyword_argument_expressions"]
        first_expression = str(
            keyword_expressions.get("text", args[0] if args else "")
        )
        second_expression = str(
            keyword_expressions.get("text_pair", args[1] if len(args) >= 2 else "")
        )
        combined = " ".join(str(item) for item in args)
        claim_present = _contains_role(first_expression, "claim")
        unit_present = _contains_role(second_expression, "unit_text") or (
            not second_expression and _contains_role(combined, "unit_text")
        )
        if claim_present and unit_present:
            pair_candidates.append(call)
    source_precedence = {
        "MDU/scripts/clip12_phase3a_final_fit/clip12p3a_final_fit.py": 0,
        "MDU/scripts/clip12_phase3_common/clip12p3_model.py": 1,
        "MDU/scripts/clip12_phase4a_inference_handoff/clip12p4a_engine.py": 2,
    }
    pair_candidates.sort(
        key=lambda item: (
            source_precedence.get(str(item["file"]), len(source_precedence)),
            int(item["line_start"]),
        )
    )
    selected = pair_candidates[0] if pair_candidates else None
    model_calls = [item for analyzer in analyzers for item in analyzer.model_calls]
    model_classes = [
        name
        for analyzer in analyzers
        for name in analyzer.class_names
        if name == "MDUSelectorVerifier"
    ]
    model_class = "MDUSelectorVerifier" if model_calls or model_classes else UNKNOWN
    tokenizer_preparation = [
        item for analyzer in analyzers for item in analyzer.tokenizer_preparation
    ]
    config_max_length = _recursive_config_value(config, "max_length")
    if selected is None:
        style = UNKNOWN
        claim_serialized = False
        unit_serialized = False
        unit_expression = ""
        keywords: Mapping[str, Any] = {}
    else:
        args = selected["positional_arguments"]
        keyword_expressions = selected["keyword_argument_expressions"]
        first_expression = str(
            keyword_expressions.get("text", args[0] if args else "")
        )
        second_expression = str(
            keyword_expressions.get("text_pair", args[1] if len(args) >= 2 else "")
        )
        if second_expression:
            style = "TOKENIZER_SEQUENCE_PAIR"
        elif re.search(r"\bzip\s*\(", first_expression) or re.search(
            r"\([^()]*(?:claim)[^()]*,[^()]*(?:unit|candidate)[^()]*\)",
            first_expression,
            flags=re.IGNORECASE,
        ):
            style = "TOKENIZER_SEQUENCE_PAIR_BATCH"
        else:
            style = "MANUAL_CONCATENATION_SINGLE_SEQUENCE"
        claim_serialized = _contains_role(first_expression, "claim")
        unit_expression = second_expression or first_expression
        unit_serialized = _contains_role(unit_expression, "unit_text")
        keywords = selected["keyword_arguments"]
    expression_lower = unit_expression.casefold()
    unit_type_serialized = any(
        marker in expression_lower
        for marker in ("unit_type", "source_type", "unittype", "sourcetype")
    )
    modality_serialized = "modality" in expression_lower
    dataset_serialized = "dataset" in expression_lower
    unit_id_serialized = any(
        marker in expression_lower for marker in ("unit_id", "unitid")
    )
    explicit = unit_type_serialized or modality_serialized or bool(
        re.search(r"\b(?:ocr|transcript|unit[_ -]?type|modality)\b", expression_lower)
    )
    encoding_verified = bool(
        selected is not None
        and model_class != UNKNOWN
        and claim_serialized
        and unit_serialized
        and style != UNKNOWN
    )
    unit_modality_encoding = (
        ("EXPLICIT" if explicit else "IMPLICIT_ONLY")
        if encoding_verified
        else UNKNOWN
    )
    truncation_expression = keywords.get("truncation")
    truncation = truncation_expression
    config_truncation = _recursive_config_value(config, "truncation")
    if isinstance(truncation, str) and config_truncation is not None:
        truncation = config_truncation
    if style == "TOKENIZER_SEQUENCE_PAIR" and truncation is True:
        truncation_strategy = "longest_first"
        truncatable_side = "both sequences as needed"
    elif truncation == "only_first":
        truncation_strategy = "only_first"
        truncatable_side = "claim sequence only"
    elif truncation == "only_second":
        truncation_strategy = "only_second"
        truncatable_side = "unit sequence only"
    else:
        truncation_strategy = truncation if truncation is not None else UNKNOWN
        truncatable_side = UNKNOWN
    max_length_expression = keywords.get("max_length")
    max_length = max_length_expression
    if not isinstance(max_length, int):
        max_length = config_max_length if isinstance(config_max_length, int) else max_length
    inspected = []
    if selected is not None:
        inspected.append(
            {
                "file": selected["file"],
                "line_start": selected["line_start"],
                "line_end": selected["line_end"],
                "scope": selected["scope"],
                "evidence": selected["call"],
            }
        )
    candidate_evidence = [
        {
            "file": item["file"],
            "line_start": item["line_start"],
            "line_end": item["line_end"],
            "scope": item["scope"],
            "evidence": item["call"],
        }
        for item in pair_candidates
    ]
    model_evidence = [
        {
            "file": item["file"],
            "line_start": item["line_start"],
            "line_end": item["line_end"],
            "scope": item["scope"],
            "evidence": item["call"],
        }
        for item in model_calls
    ]
    padding_expression = keywords.get("padding")
    padding = padding_expression
    config_padding = _recursive_config_value(config, "padding")
    if isinstance(padding, str) and config_padding is not None:
        padding = config_padding
    return {
        "source_files_inspected": [item.as_posix() for item in _SOURCE_RELATIVE_PATHS],
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "model_class_used": model_class,
        "tokenizer_preparation_path": tokenizer_preparation,
        "encoding_contract_evidence": inspected,
        "encoding_contract_candidates": candidate_evidence,
        "encoding_contract_candidate_count": len(candidate_evidence),
        "model_class_evidence": model_evidence,
        "encoding_contract_verified": encoding_verified,
        "runtime_trace_available": False,
        "runtime_trace_reason": (
            "Authoritative modules are not imported because their import path may "
            "construct neural dependencies; static AST evidence is used instead."
        ),
        "claim_unit_encoding_style": style,
        "claim_text_serialized": claim_serialized,
        "unit_text_serialized": unit_serialized,
        "unit_type_serialized": unit_type_serialized,
        "modality_serialized": modality_serialized,
        "dataset_serialized": dataset_serialized,
        "unit_id_serialized": unit_id_serialized,
        "unit_fixed_prefix_inserted": explicit,
        "unit_serialization_expression": unit_expression,
        "tokenizer_keyword_arguments": dict(keywords),
        "maximum_sequence_length_expression": max_length_expression,
        "maximum_sequence_length": max_length,
        "truncation_expression": truncation_expression,
        "truncation_strategy": truncation_strategy,
        "truncatable_side": truncatable_side,
        "padding_expression": padding_expression,
        "padding": padding if padding is not None else UNKNOWN,
        "unit_modality_encoding": unit_modality_encoding,
    }


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


def _write_bytes(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    return _write_bytes(path, _json_bytes(payload))


def _audit_readme(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Step 2.6R-1C Template / Modality Shortcut Audit",
            "",
            "This is a read-only audit. It does not train or modify Frozen G1,",
            "the selector/veracity heads, calibration data, or production code.",
            "",
            f"- Shortcut risk: {report['shortcut_risk_classification']}",
            f"- Training recommendation: {report['training_authorization_recommendation']}",
            f"- Claim-only modality accuracy: {report['claim_only_template_modality_accuracy']}",
            f"- Unit modality encoding: {report['unit_modality_encoding']}",
            "- Template-swapped claims are diagnostic only and must not be treated as literal production evidence.",
            "- Formal Validation and Formal Test were not accessed.",
            "",
        ]
    )


def run_shortcut_audit(
    *,
    project_root: Path,
    calibration_dir: Path,
    output_dir: Path,
    expected_counts: ExpectedCounts = AUTHORITATIVE_COUNTS,
) -> Mapping[str, Any]:
    output_dir = _reject_formal_path(output_dir, "output directory")
    artifacts = load_calibration_artifacts(calibration_dir)
    before_hashes = {
        name: sha256_file(artifacts.calibration_dir / name)
        for name in _REQUIRED_CALIBRATION_FILES
    }
    examples = artifacts.examples
    template_report, control_manifest = build_template_analysis(examples)
    actual_counts = ExpectedCounts(
        total=template_report["total_example_count"],
        ocr=template_report["ocr_example_count"],
        transcript=template_report["transcript_example_count"],
    )
    if actual_counts != expected_counts:
        raise AuditInputError(
            "calibration example counts do not match the registered authoritative counts"
        )
    if artifacts.build_report.get("calibration_train_example_count") != len(
        artifacts.train_examples
    ) or artifacts.build_report.get("calibration_dev_example_count") != len(
        artifacts.dev_examples
    ):
        raise AuditInputError("calibration counts disagree with build report")
    if (
        artifacts.build_report.get("ocr_example_count") != actual_counts.ocr
        or artifacts.build_report.get("transcript_example_count")
        != actual_counts.transcript
    ):
        raise AuditInputError("calibration modality counts disagree with build report")
    encoding_contract = inspect_encoding_contract(project_root)
    shortcut_risk = classify_shortcut_risk(
        float(template_report["claim_only_template_modality_accuracy"]),
        str(encoding_contract["unit_modality_encoding"]),
        encoding_contract_verified=bool(
            encoding_contract["encoding_contract_verified"]
        ),
    )
    recommendation = recommend_training_action(shortcut_risk)
    report: Dict[str, Any] = {
        "status": "COMPLETED",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "calibration_train_sha256": artifacts.file_sha256[
            "calibration_train.jsonl"
        ],
        "calibration_dev_sha256": artifacts.file_sha256["calibration_dev.jsonl"],
        **encoding_contract,
        **template_report,
        "shortcut_risk_classification": shortcut_risk,
        "training_authorization_recommendation": recommendation,
        "selection_outputs_inspected": False,
        "veracity_labels_inspected": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "training_started": False,
        "production_or_model_code_changed": False,
        "frozen_g1_checkpoint_unchanged": True,
        "encoder_unchanged": True,
        "veracity_head_unchanged": True,
        "selection_head_unchanged": True,
        "candidate_exposure_unchanged": True,
        "sample_pooling_unchanged": True,
        "top_k_explanation_only_boundary_unchanged": True,
    }
    calibration_resolved = artifacts.calibration_dir
    if (
        output_dir == calibration_resolved
        or output_dir in calibration_resolved.parents
        or calibration_resolved in output_dir.parents
    ):
        raise AuditInputError("output directory must be isolated from calibration artifacts")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AuditInputError("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "encoding_contract.json", encoding_contract)
    _write_json(output_dir / "template_leakage_report.json", template_report)
    manifest_bytes = _jsonl_bytes(control_manifest)
    manifest_sha = _write_bytes(
        output_dir / "template_control_manifest.jsonl", manifest_bytes
    )
    (output_dir / "template_control_manifest.sha256").write_text(
        manifest_sha + "\n", encoding="utf-8"
    )
    _write_json(output_dir / "shortcut_audit_report.json", report)
    (output_dir / "README_AUDIT.md").write_text(
        _audit_readme(report), encoding="utf-8"
    )
    after_hashes = {
        name: sha256_file(artifacts.calibration_dir / name)
        for name in _REQUIRED_CALIBRATION_FILES
    }
    if before_hashes != after_hashes:
        raise AuditInputError("authoritative calibration artifacts changed during audit")
    return report
