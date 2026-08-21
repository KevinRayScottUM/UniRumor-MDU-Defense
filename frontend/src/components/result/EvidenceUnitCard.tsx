import type { PublicEvidenceUnit } from "../../types";
import { Badge, Card } from "../ui";

export type EvidenceUnitVariant = "candidate" | "selected" | "supplemental";

export interface EvidenceUnitCardProps {
  unit: PublicEvidenceUnit;
  variant?: EvidenceUnitVariant;
}

const SOURCE_LABELS: Record<PublicEvidenceUnit["source_type"], string> = {
  text: "Text",
  transcript: "Transcript",
  ocr: "OCR",
  visual_observation: "Visual observation",
};

function numericValue(value: number): string {
  return String(value);
}

function timeRange(unit: PublicEvidenceUnit): string | undefined {
  if (unit.start_time !== null && unit.end_time !== null) {
    return `${numericValue(unit.start_time)}–${numericValue(unit.end_time)} s`;
  }
  if (unit.start_time !== null) return `${numericValue(unit.start_time)} s`;
  if (unit.end_time !== null) return `Ends at ${numericValue(unit.end_time)} s`;
  return undefined;
}

function optionalMetadata(unit: PublicEvidenceUnit) {
  return [
    timeRange(unit) ? { label: "Time", value: timeRange(unit) as string } : null,
    unit.frame_id ? { label: "Frame", value: unit.frame_id } : null,
    unit.frame_ids.length > 0
      ? { label: "Frames", value: unit.frame_ids.join(", ") }
      : null,
    unit.confidence !== null
      ? { label: "Confidence", value: numericValue(unit.confidence) }
      : null,
    unit.source_index !== null
      ? { label: "Source index", value: String(unit.source_index) }
      : null,
    unit.observation_type
      ? { label: "Observation", value: unit.observation_type }
      : null,
    unit.bbox !== null
      ? { label: "Bounding box", value: unit.bbox.map(numericValue).join(", ") }
      : null,
  ].filter((item): item is { label: string; value: string } => item !== null);
}

export function EvidenceUnitCard({
  unit,
  variant = "candidate",
}: EvidenceUnitCardProps) {
  const metadata = optionalMetadata(unit);
  const selectionLabel =
    unit.selection_score === null
      ? undefined
      : `Selection score ${numericValue(unit.selection_score)}`;

  return (
    <Card
      className={`evidence-unit evidence-unit--${variant}`}
      variant={variant === "selected" ? "raised" : "default"}
    >
      <div className="evidence-unit__topline">
        <div className="evidence-unit__badges">
          <Badge className={`evidence-source evidence-source--${unit.source_type}`}>
            {SOURCE_LABELS[unit.source_type]}
          </Badge>
          {variant === "selected" ? (
            <Badge tone="success">Selected explanation</Badge>
          ) : null}
          {variant === "supplemental" ? (
            <Badge tone="research">Supplemental</Badge>
          ) : null}
        </div>
        {selectionLabel ? (
          <span className="evidence-unit__selection">{selectionLabel}</span>
        ) : null}
      </div>

      <div className="evidence-unit__identity">
        <p>Unit identifier</p>
        <code>{unit.unit_id}</code>
      </div>
      <blockquote>{unit.text}</blockquote>

      {metadata.length > 0 ? (
        <dl className="evidence-unit__metadata">
          {metadata.map((item) => (
            <div key={item.label}>
              <dt>{item.label}</dt>
              <dd>{item.value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="evidence-unit__metadata-empty">
          No additional public metadata was provided for this unit.
        </p>
      )}

      <div className="evidence-unit__provenance">
        <div>
          <span>Producer</span>
          <strong>{unit.producer}</strong>
        </div>
        <div>
          <span>Extraction</span>
          <strong>{unit.extraction_method}</strong>
        </div>
      </div>

      {unit.evidence_refs.length > 0 || unit.source_unit_ids.length > 0 ? (
        <div className="evidence-unit__references">
          {unit.evidence_refs.length > 0 ? (
            <p>
              <span>Evidence references</span>
              {unit.evidence_refs.join(", ")}
            </p>
          ) : null}
          {unit.source_unit_ids.length > 0 ? (
            <p>
              <span>Source units</span>
              {unit.source_unit_ids.join(", ")}
            </p>
          ) : null}
        </div>
      ) : null}
    </Card>
  );
}
