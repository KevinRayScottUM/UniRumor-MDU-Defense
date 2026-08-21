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

        <Card className="research-brief" variant="glass">
          <div className="research-brief__topline">
            <span>Research interface</span>
            <Badge tone="neutral">Public API v1</Badge>
          </div>
          <h2>From a focal claim to auditable evidence.</h2>
          <p>
            The interface keeps request handling, production execution, and
            result presentation visually distinct.
          </p>
          <ol className="research-brief__flow" aria-label="Verification workflow">
            <li>
              <span>01</span>
              Claim + video
            </li>
            <li>
              <span>02</span>
              Managed job
            </li>
            <li>
              <span>03</span>
              Public result
            </li>
          </ol>
          <dl className="research-brief__facts">
            <div>
              <dt>Validation</dt>
              <dd>Server enforced</dd>
            </div>
            <div>
              <dt>Scientific output</dt>
              <dd>Never recomputed</dd>
            </div>
          </dl>
          <Link className="research-brief__demo-link" to="/demo">
            View illustrative result demo
            <span aria-hidden="true">→</span>
          </Link>
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
