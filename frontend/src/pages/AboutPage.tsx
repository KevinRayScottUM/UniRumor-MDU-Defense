import { Link } from "react-router-dom";

import { MDUConceptDiagram, VerificationWorkflow } from "../components/research";
import { Badge, Card, SectionHeader } from "../components/ui";

const mduConcepts = [
  {
    index: "01",
    title: "Claim-centered verification",
    description:
      "Every session begins with one exact focal claim. That claim remains the stable reference for the request and public result.",
  },
  {
    index: "02",
    title: "Candidate units",
    description:
      "The public evidence view separates inspectable text, transcript, and OCR units instead of presenting one undifferentiated media summary.",
  },
  {
    index: "03",
    title: "Evidence selection",
    description:
      "Selection scores order explanation units only. They do not determine which evaluated eligible units contribute to the sample verdict.",
  },
  {
    index: "04",
    title: "Explainable verification",
    description:
      "A completed result keeps the final verdict, full eligible candidate exposure, and selected explanation references visibly distinct.",
  },
] as const;

export function AboutPage() {
  return (
    <div className="about-mdu-page">
      <section aria-labelledby="about-mdu-heading" className="about-mdu-hero">
        <div className="about-mdu-hero__copy">
          <Badge tone="research" withDot>
            Method overview
          </Badge>
          <SectionHeader
            className="about-mdu-hero__header"
            description="UniRumor-MDU organizes multimodal verification around discrete, inspectable evidence units linked to one focal claim. The interface exposes those public units without recreating the scientific decision process."
            eyebrow="About the research interface"
            headingId="about-mdu-heading"
            headingLevel={1}
            title="What is a Minimal Deceptive Unit?"
          />
        </div>

        <Card className="about-mdu-definition" variant="raised">
          <span className="about-mdu-definition__mark" aria-hidden="true">
            MDU
          </span>
          <p>
            In this system, an MDU is an evidence unit presented for claim-level
            inspection and explanation. Candidate units preserve their source and
            metadata so a result can be reviewed beyond its final label.
          </p>
          <small>
            Unit-level presentation does not imply that every candidate is independently decisive.
          </small>
        </Card>
      </section>

      <section aria-labelledby="mdu-principles-heading" className="mdu-principles">
        <SectionHeader
          description="The website follows the same separation of responsibilities as the public production contract."
          eyebrow="Core concepts"
          headingId="mdu-principles-heading"
          title="Evidence that remains traceable"
        />
        <div className="mdu-principles__grid">
          {mduConcepts.map((concept) => (
            <Card className="mdu-principle" key={concept.index}>
              <span>{concept.index}</span>
              <h3>{concept.title}</h3>
              <p>{concept.description}</p>
            </Card>
          ))}
        </div>
      </section>

      <section aria-labelledby="mdu-model-heading" className="mdu-model-section">
        <SectionHeader
          description="The diagram names structural roles only. It contains no uploaded media, model output, or fabricated evidence."
          eyebrow="Concept illustration"
          headingId="mdu-model-heading"
          title="How evidence becomes reviewable"
        />
        <MDUConceptDiagram />
      </section>

      <section aria-labelledby="method-flow-heading" className="about-method-flow">
        <SectionHeader
          description="The interface reveals each public boundary while the production runtime remains the sole owner of scientific execution."
          eyebrow="Verification flow"
          headingId="method-flow-heading"
          title="A closed runtime, an open explanation"
        />
        <VerificationWorkflow />
      </section>

      <aside className="about-mdu-boundary" aria-labelledby="boundary-heading">
        <div>
          <p>Scientific boundary</p>
          <h2 id="boundary-heading">The website presents; it does not predict.</h2>
        </div>
        <p>
          Verdicts, evidence sufficiency, candidate units, and explanation IDs come
          from the backend response. The frontend does not run models, calculate a
          verdict, or turn missing evidence into a prediction.
        </p>
        <Link className="home-primary-link" to="/#verify">
          Start a verification
          <span aria-hidden="true">→</span>
        </Link>
      </aside>
    </div>
  );
}
