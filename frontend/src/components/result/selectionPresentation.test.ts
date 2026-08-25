import { describe, expect, it } from "vitest";

import type { PublicEvidenceUnit } from "../../types";
import { deriveSelectionPresentation } from "./selectionPresentation";

function candidate(
  unitId: string,
  selectionScore: number | null,
): PublicEvidenceUnit {
  return {
    unit_id: unitId,
    source_type: "transcript",
    text: unitId,
    start_time: null,
    end_time: null,
    frame_id: null,
    bbox: null,
    confidence: null,
    producer: "test",
    eligible_for_frozen_g1: true,
    selection_score: selectionScore,
    logits: null,
    extraction_method: "test",
    source_index: null,
    frame_ids: [],
    evidence_refs: [],
    source_unit_ids: [],
    observation_type: null,
  };
}

describe("selection presentation metadata", () => {
  it("derives ranks without mutating authoritative candidate exposure order", () => {
    const candidates = [
      candidate("unit-low", -0.4),
      candidate("unit-high", 0.8),
      candidate("unit-middle", -0.1),
    ];
    const originalOrder = candidates.map((unit) => unit.unit_id);

    const presentation = deriveSelectionPresentation(candidates, ["unit-low"]);

    expect(candidates.map((unit) => unit.unit_id)).toEqual(originalOrder);
    expect(presentation.map((item) => item.selectionRank)).toEqual([3, 1, 2]);
    expect(presentation.map((item) => item.selectionRankTotal)).toEqual([3, 3, 3]);
  });

  it("uses exposure order as the deterministic score tie-break", () => {
    const presentation = deriveSelectionPresentation(
      [candidate("unit-first", 0.2), candidate("unit-second", 0.2)],
      [],
    );

    expect(presentation.map((item) => item.selectionRank)).toEqual([1, 2]);
  });

  it("leaves null and non-finite scores unranked while ranking negatives normally", () => {
    const presentation = deriveSelectionPresentation(
      [
        candidate("unit-null", null),
        candidate("unit-negative-low", -0.9),
        candidate("unit-nan", Number.NaN),
        candidate("unit-negative-high", -0.2),
      ],
      [],
    );

    expect(presentation.map((item) => item.selectionRank)).toEqual([
      null,
      2,
      null,
      1,
    ]);
  });

  it("uses backend-selected IDs for membership and display order metadata", () => {
    const presentation = deriveSelectionPresentation(
      [
        candidate("unit-highest", 0.9),
        candidate("unit-backend-first", 0.1),
        candidate("unit-backend-second", 0.2),
      ],
      ["unit-backend-first", "unit-backend-second"],
    );

    expect(presentation.map((item) => item.topKSelected)).toEqual([
      false,
      true,
      true,
    ]);
    expect(presentation.map((item) => item.topKDisplayRank)).toEqual([
      null,
      1,
      2,
    ]);
  });
});
