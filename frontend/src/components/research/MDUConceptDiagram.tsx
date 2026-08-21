import { Badge, Card } from "../ui";

const candidateUnits = [
  { label: "Text unit", source: "Text", selected: false },
  { label: "Transcript unit", source: "Transcript", selected: true },
  { label: "OCR unit", source: "OCR", selected: true },
] as const;

export function MDUConceptDiagram() {
  return (
    <Card className="mdu-concept" variant="glass">
      <div className="mdu-concept__header">
        <div>
          <p>Conceptual model</p>
          <h3>From one focal claim to inspectable evidence</h3>
        </div>
        <Badge tone="neutral">No result data</Badge>
      </div>

      <div
        aria-label="Conceptual flow from a focal claim through candidate units to selected explanation units"
        className="mdu-concept__canvas"
      >
        <div className="mdu-concept__claim">
          <span>01 · Focal claim</span>
          <strong>The exact statement under verification</strong>
        </div>

        <div aria-hidden="true" className="mdu-concept__connector">
          <span />
          <b>Candidate construction</b>
        </div>

        <div className="mdu-concept__candidates">
          <div className="mdu-concept__stage-label">
            <span>02</span>
            Candidate units
          </div>
          <div className="mdu-concept__unit-list">
            {candidateUnits.map((unit) => (
              <div
                className={`mdu-concept__unit${unit.selected ? " mdu-concept__unit--selected" : ""}`}
                key={unit.label}
              >
                <span>{unit.source}</span>
                <strong>{unit.label}</strong>
                {unit.selected ? <small>Selected for explanation</small> : null}
              </div>
            ))}
          </div>
        </div>

        <div aria-hidden="true" className="mdu-concept__connector">
          <span />
          <b>Explanation selection</b>
        </div>

        <div className="mdu-concept__explanation">
          <span>03 · Public explanation</span>
          <strong>Ordered unit references</strong>
          <p>Selection identifies explanation units; it does not replace the full eligible prediction pool.</p>
        </div>
      </div>
    </Card>
  );
}
