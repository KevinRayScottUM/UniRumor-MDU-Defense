"""Independent deterministic A/B blank packets and private reconstruction keys."""

import csv
import hashlib
import io

from .artifacts import compact
from .schemas import CohortError, PUBLIC_COLUMNS, REVIEWER_SALTS


def key(reviewer, purpose, *values):
    return hashlib.sha256(compact([REVIEWER_SALTS[reviewer], purpose, *values])).hexdigest()


def packets(selected):
    outputs, mappings, orders = {}, {}, {}
    expected = {(row.case.canonical_case_id, unit[0]) for row in selected for unit in row.case.candidates}
    for reviewer in REVIEWER_SALTS:
        rows, private, underlying = [], [], []
        ordered = sorted(selected, key=lambda row: (key(reviewer, "case-order", row.case.canonical_case_id),
                                                   row.case.canonical_case_id))
        for row in ordered:
            case = row.case
            case_id = "case-" + key(reviewer, "case-id", case.canonical_case_id)
            candidates = sorted(enumerate(case.candidates), key=lambda item:
                                (key(reviewer, "unit-order", case.canonical_case_id, item[1][0]), item[1][0]))
            for position, unit in candidates:
                unit_id = "unit-" + key(reviewer, "unit-id", case.canonical_case_id, unit[0])
                rows.append(dict(zip(PUBLIC_COLUMNS, (case_id, case.claim, unit_id, unit[3], "", "", ""))))
                private.append({"reviewer": reviewer, "review_case_id": case_id,
                                "review_unit_id": unit_id, "dataset": case.dataset,
                                "canonical_case_id": case.canonical_case_id,
                                "original_case_id": case.original_case_id, "repair_split": row.repair_split,
                                "unit_id": unit[0], "original_candidate_position": position})
                underlying.append((case.canonical_case_id, unit[0]))
        if len(underlying) != len(expected) or set(underlying) != expected:
            raise CohortError("review packet candidate coverage mismatch")
        if len({r["review_unit_id"] for r in rows}) != len(rows):
            raise CohortError("review pseudonym collision")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=PUBLIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        outputs[f"reviewer_{reviewer}_template.csv"] = stream.getvalue().encode("utf-8")
        mappings[reviewer] = private
        orders[reviewer] = underlying
    if orders["A"] == orders["B"]:
        raise CohortError("independent A/B ordering unexpectedly identical")
    case_orders = {reviewer: list(dict.fromkeys(case for case, _ in order))
                   for reviewer, order in orders.items()}
    if case_orders["A"] == case_orders["B"]:
        raise CohortError("independent A/B case ordering unexpectedly identical")
    if {r["review_case_id"] for r in mappings["A"]} & {r["review_case_id"] for r in mappings["B"]}:
        raise CohortError("A/B case pseudonyms are not independent")
    if {r["review_unit_id"] for r in mappings["A"]} & {r["review_unit_id"] for r in mappings["B"]}:
        raise CohortError("A/B mapping keys are not independent")
    return outputs, {"status": "PRIVATE_FROZEN_MAPPING", "reviewer_salts": REVIEWER_SALTS,
                     "reviewer_A": mappings["A"], "reviewer_B": mappings["B"]}
