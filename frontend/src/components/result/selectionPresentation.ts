import type { PublicEvidenceUnit } from "../../types";

export interface SelectionPresentationMetadata {
  selectionRank: number | null;
  selectionRankTotal: number;
  topKSelected: boolean;
  topKDisplayRank: number | null;
}

export function deriveSelectionPresentation(
  candidates: readonly PublicEvidenceUnit[],
  selectedIds: readonly string[],
): SelectionPresentationMetadata[] {
  const scoredCandidates = candidates
    .map((unit, exposureIndex) => ({
      exposureIndex,
      score: unit.selection_score,
    }))
    .filter(
      (candidate): candidate is { exposureIndex: number; score: number } =>
        typeof candidate.score === "number" && Number.isFinite(candidate.score),
    )
    .sort(
      (left, right) =>
        right.score - left.score || left.exposureIndex - right.exposureIndex,
    );

  const rankByExposureIndex = new Map<number, number>();
  scoredCandidates.forEach((candidate, index) => {
    rankByExposureIndex.set(candidate.exposureIndex, index + 1);
  });

  const selectedRankById = new Map<string, number>();
  selectedIds.forEach((unitId, index) => {
    if (!selectedRankById.has(unitId)) {
      selectedRankById.set(unitId, index + 1);
    }
  });

  return candidates.map((unit, exposureIndex) => {
    const topKDisplayRank = selectedRankById.get(unit.unit_id) ?? null;
    return {
      selectionRank: rankByExposureIndex.get(exposureIndex) ?? null,
      selectionRankTotal: candidates.length,
      topKSelected: topKDisplayRank !== null,
      topKDisplayRank,
    };
  });
}
