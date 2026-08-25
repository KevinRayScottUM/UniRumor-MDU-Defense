import type { PublicEvidenceUnit } from "../../types";
import { Badge, Card } from "../ui";
import { EvidenceFrameGallery } from "./EvidenceFrameGallery";

export type EvidenceUnitVariant = "candidate" | "selected" | "supplemental";

export interface EvidenceUnitCardProps {
  unit: PublicEvidenceUnit;
  variant?: EvidenceUnitVariant;
  jobId?: string;
  onVisualXAIReady?: () => void;
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

function groundedExplanation(unit: PublicEvidenceUnit): string {
  const frames = unit.evidence_frames ?? [];
  const regionCount = frames.reduce(
    (count, frame) => count + frame.regions.length,
    0,
  );
  if (unit.source_type === "ocr") {
    if (regionCount > 0) {
      return `The recognized text is grounded in ${String(regionCount)} recorded OCR region${regionCount === 1 ? "" : "s"}. Highlighted boxes reproduce backend coordinates without changing the text.`;
    }
    return "The OCR unit has frame-level provenance, but no localized OCR region was provided. The interface does not invent coordinates.";
  }
  if (unit.source_type === "visual_observation") {
    if (frames.length > 0) {
      const hasAttribution = frames.some(
        (frame) => frame.xai?.status === "available",
      );
      return hasAttribution
        ? "The observation is grounded in the exact observer source frame. Highlighting is model-derived post-hoc occlusion attribution, not face recognition, object identity, or causal attention."
        : "The observation is grounded in the referenced public frames. No visual region or identity is invented when model-derived attribution is unavailable.";
    }
    return "The observation has public provenance metadata, but no frame imagery was provided. No visual region or identity is inferred.";
  }
  return "This unit is presented from the authoritative textual evidence and provenance returned by the backend.";
}

export function EvidenceUnitCard({
  unit,
  variant = "candidate",
  jobId,
  onVisualXAIReady,
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

      <section className="evidence-unit__summary" aria-label="Evidence summary">
        <div className="evidence-unit__section-heading">
          <p>Evidence summary</p>
          <h4>Authoritative candidate content</h4>
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
      </section>

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

      {unit.source_type === "ocr" || unit.source_type === "visual_observation" ? (
        <EvidenceFrameGallery
          frames={unit.evidence_frames ?? []}
          jobId={jobId}
          onVisualXAIReady={onVisualXAIReady}
          sourceType={unit.source_type}
          unitId={unit.unit_id}
        />
      ) : null}

      <section className="evidence-unit__grounding" aria-label="Grounded explanation">
        <p>Grounded explanation</p>
        <h4>Why this evidence is inspectable</h4>
        <span>{groundedExplanation(unit)}</span>
      </section>
    </Card>
  );
}
