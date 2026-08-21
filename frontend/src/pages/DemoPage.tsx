import { Link } from "react-router-dom";

import { Badge, Card, SectionHeader } from "../components/ui";

const illustrativeUnits = [
  {
    id: "demo_text_01",
    source: "Text",
    text: "Example article text would appear here as a candidate unit.",
    selected: false,
  },
  {
    id: "demo_transcript_02",
    source: "Transcript",
    text: "Example timestamped transcript content would appear here.",
    selected: true,
  },
  {
    id: "demo_ocr_03",
    source: "OCR",
    text: "Example visible on-screen text would appear here.",
    selected: true,
  },
] as const;

export function DemoPage() {
  const selectedUnits = illustrativeUnits.filter((unit) => unit.selected);

  return (
    <div className="demo-page">
      <section aria-labelledby="demo-heading" className="demo-hero">
        <Badge tone="research" withDot>
          Static walkthrough
        </Badge>
        <SectionHeader
          className="demo-hero__header"
          description="Explore the information hierarchy and evidence-unit presentation without submitting media or contacting the production API."
          eyebrow="Research demo preview"
          headingId="demo-heading"
          headingLevel={1}
          title="See how an explainable result is organized"
        />
      </section>

      <aside className="demo-disclosure" role="note">
        <span aria-hidden="true">i</span>
        <div>
          <strong>Illustrative demo only. Not a live model result.</strong>
          <p>
            All claim and unit text below is interface copy. It is not extracted
            evidence, a prediction, or a scientific finding.
          </p>
        </div>
      </aside>

      <section aria-labelledby="demo-result-heading" className="demo-result">
        <div className="demo-result__heading">
          <p>Example result surface</p>
          <h2 id="demo-result-heading">Claim and verdict hierarchy</h2>
        </div>
        <div className="demo-result__overview">
          <Card className="demo-claim" variant="glass">
            <p>Example claim</p>
            <blockquote>
              “This video was recorded at the location named in its caption.”
            </blockquote>
            <small>Neutral example wording for layout demonstration only.</small>
          </Card>

          <Card
            aria-label="Illustrative FAKE verdict presentation; not a live result"
            className="demo-verdict"
            variant="raised"
          >
            <p>Example verdict presentation</p>
            <strong>FAKE</strong>
            <Badge tone="warning">Illustrative label</Badge>
            <small>This label does not describe the example claim.</small>
          </Card>
        </div>
      </section>

      <section aria-labelledby="demo-units-heading" className="demo-evidence">
        <div className="demo-result__heading demo-result__heading--split">
          <div>
            <p>Example MDU explanation</p>
            <h2 id="demo-units-heading">Candidate and selected units</h2>
          </div>
          <Badge tone="neutral">Example structure</Badge>
        </div>

        <div className="demo-evidence__flow">
          <section aria-labelledby="demo-candidates-heading" className="demo-stage">
            <div className="demo-stage__header">
              <span>01</span>
              <div>
                <p>Public evidence view</p>
                <h3 id="demo-candidates-heading">Candidate units</h3>
              </div>
            </div>
            <ul className="demo-unit-list">
              {illustrativeUnits.map((unit) => (
                <li className="demo-unit" key={unit.id}>
                  <div>
                    <Badge tone="neutral">{unit.source}</Badge>
                    <code>{unit.id}</code>
                  </div>
                  <p>{unit.text}</p>
                </li>
              ))}
            </ul>
          </section>

          <div aria-hidden="true" className="demo-flow-connector">
            <span />
            <p>Explanation references</p>
            <strong>→</strong>
          </div>

          <section aria-labelledby="demo-selected-heading" className="demo-stage demo-stage--selected">
            <div className="demo-stage__header">
              <span>02</span>
              <div>
                <p>Ordered explanation view</p>
                <h3 id="demo-selected-heading">Selected units</h3>
              </div>
            </div>
            <ol className="demo-unit-list">
              {selectedUnits.map((unit) => (
                <li className="demo-unit demo-unit--selected" key={unit.id}>
                  <div>
                    <Badge tone="success">Selected example</Badge>
                    <code>{unit.id}</code>
                  </div>
                  <p>{unit.text}</p>
                </li>
              ))}
            </ol>
          </section>
        </div>

        <p className="demo-evidence__boundary">
          In a live completed session, claim text, verdict, units, metadata, and
          selected IDs are rendered only from the backend response.
        </p>
      </section>

      <aside aria-labelledby="demo-next-heading" className="demo-next-step">
        <div>
          <p>Ready to use the production workflow?</p>
          <h2 id="demo-next-heading">Start with an exact claim and source video.</h2>
        </div>
        <Link className="home-primary-link" to="/#verify">
          Open verification
          <span aria-hidden="true">→</span>
        </Link>
      </aside>
    </div>
  );
}
