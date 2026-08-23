import { Link } from "react-router-dom";

import { MDUConceptDiagram, VerificationWorkflow } from "../components/research";
import { VerificationSubmissionCard } from "../components/VerificationSubmissionCard";
import { Badge, Card, SectionHeader } from "../components/ui";

const motivationPoints = [
  {
    index: "01",
    title: "Keep the claim in focus",
    description:
      "One exact focal claim anchors the request, the verification record, and the evidence review.",
  },
  {
    index: "02",
    title: "Separate evidence sources",
    description:
      "Text, transcript, OCR, and supplemental visual observations retain distinct roles in the public result.",
  },
  {
    index: "03",
    title: "Make the result inspectable",
    description:
      "Candidate units and selected explanation references remain visible beside the authoritative outcome.",
  },
] as const;

const illustrativeUnits = [
  {
    id: "01",
    type: "Title text",
    content: "48 Hour Challenge is safe and encouraged.",
    score: "High relevance",
    tone: "high",
  },
  {
    id: "02",
    type: "OCR evidence",
    content: "Do not attempt this challenge.",
    score: "Medium relevance",
    tone: "medium",
  },
  {
    id: "03",
    type: "Transcript evidence",
    content: "Experts warn users about potential risks.",
    score: "High relevance",
    tone: "high",
  },
] as const;

export function HomePage() {
  return (
    <div className="home-page">
      <section aria-labelledby="home-heading" className="home-hero">
        <div className="home-hero__copy">
          <Badge tone="research" withDot>
            Research demonstration
          </Badge>
          <SectionHeader
            className="home-hero__header"
            description="UniRumor-MDU decomposes multimedia claims into auditable evidence units for transparent verification."
            eyebrow="Multimodal disinformation understanding"
            headingId="home-heading"
            headingLevel={1}
            title="Explainable Multimodal Misinformation Verification"
          />
          <div className="home-hero__actions">
            <a className="home-primary-link" href="#verify">
              Start Verification
              <span aria-hidden="true">↓</span>
            </a>
            <p>One claim. One video. Server-authoritative output.</p>
          </div>
        </div>

        <Card
          aria-label="Illustrative Minimal Deceptive Unit preview"
          className="mdu-demo-preview"
          role="region"
          variant="glass"
        >
          <div className="mdu-demo-preview__topline">
            <span>
              <i aria-hidden="true" />
              MDU interface preview
            </span>
            <Badge tone="research">Illustrative</Badge>
          </div>

          <div className="mdu-demo-preview__claim">
            <span>Claim</span>
            <blockquote>
              “48 Hour Challenge video claims a dangerous online challenge is
              harmless.”
            </blockquote>
          </div>

          <div className="mdu-demo-preview__connector" aria-hidden="true">
            <span />
            <strong>↓</strong>
            <p>Candidate Units</p>
          </div>

          <div className="mdu-demo-preview__units">
            {illustrativeUnits.map((unit) => (
              <article className="mdu-demo-unit" key={unit.id}>
                <div className="mdu-demo-unit__header">
                  <span>Unit {unit.id}</span>
                  <small>{unit.type}</small>
                </div>
                <p>{unit.content}</p>
                <div className={`mdu-demo-unit__score mdu-demo-unit__score--${unit.tone}`}>
                  <i aria-hidden="true" />
                  {unit.score}
                </div>
              </article>
            ))}
          </div>

          <ol
            aria-label="Illustrative claim-to-verification flow"
            className="mdu-demo-preview__pipeline"
          >
            <li>
              <span>01</span>
              <strong>Claim</strong>
            </li>
            <li>
              <span>02</span>
              <strong>Candidate Units</strong>
            </li>
            <li>
              <span>03</span>
              <strong>Selection Score</strong>
            </li>
            <li>
              <span>04</span>
              <strong>Evidence Explanation</strong>
            </li>
            <li>
              <span>05</span>
              <strong>Final Verification</strong>
            </li>
          </ol>

          <div className="mdu-demo-preview__footer">
            <p>Illustrative interface only. Final results come from the backend.</p>
            <Link to="/demo">
              Explore demo <span aria-hidden="true">↗</span>
            </Link>
          </div>
        </Card>
      </section>

      <section aria-labelledby="motivation-heading" className="research-motivation">
        <div className="research-motivation__intro">
          <SectionHeader
            description="A sample-level verdict answers what the system returned. Unit-level evidence helps show what public information is available for review."
            eyebrow="Research motivation"
            headingId="motivation-heading"
            title="Move from a label to an evidence trail"
          />
          <Link className="research-text-link" to="/about">
            Explore the MDU approach
            <span aria-hidden="true">→</span>
          </Link>
        </div>
        <div className="research-motivation__points">
          {motivationPoints.map((point) => (
            <article key={point.index}>
              <span>{point.index}</span>
              <div>
                <h3>{point.title}</h3>
                <p>{point.description}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section
        aria-labelledby="verification-heading"
        className="verification-section"
        id="verify"
      >
        <SectionHeader
          className="verification-section__header"
          description="Enter the claim exactly as stated and attach its source video. The backend validates the complete request before admitting a job."
          eyebrow="Start a verification"
          headingId="verification-heading"
          title="Submit a claim and video"
        />
        <VerificationSubmissionCard />
      </section>

      <section
        aria-labelledby="workflow-heading"
        className="workflow-showcase"
        id="sessions"
      >
        <SectionHeader
          description="Submission creates a server-owned session. The website follows its authoritative state and reveals the public result only after completion."
          eyebrow="Verification sessions"
          headingId="workflow-heading"
          title="From request to reviewable evidence"
        />
        <VerificationWorkflow />
        <p className="workflow-showcase__note">
          Session pages reflect accepted, queued, running, completed, failed, and
          expired backend states. The website does not create synthetic progress or
          persistent history.
        </p>
      </section>

      <section aria-labelledby="mdu-preview-heading" className="mdu-preview">
        <div className="mdu-preview__header">
          <SectionHeader
            description="The MDU view organizes public evidence into candidate units and ordered explanation references while keeping the final verdict authoritative to the backend."
            eyebrow="Minimal Deceptive Unit"
            headingId="mdu-preview-heading"
            title="A unit-level view of verification"
          />
          <Link className="research-text-link" to="/about">
            Read about MDU
            <span aria-hidden="true">→</span>
          </Link>
        </div>
        <MDUConceptDiagram />
      </section>

      <aside aria-labelledby="research-boundary-heading" className="about-band">
        <div className="about-band__line" aria-hidden="true" />
        <div>
          <p className="about-band__eyebrow">Research boundary</p>
          <h2 id="research-boundary-heading">
            Interface clarity without scientific reinterpretation
          </h2>
        </div>
        <p>
          The frontend submits public inputs and renders public responses only. It
          does not access datasets, execute models, infer progress, or calculate
          verification outcomes.
        </p>
      </aside>
    </div>
  );
}
