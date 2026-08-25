import { Link, useParams } from "react-router-dom";

import { EvidenceUnitCard } from "../components/result";
import { deriveSelectionPresentation } from "../components/result/selectionPresentation";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorMessage,
  LoadingState,
  SectionHeader,
} from "../components/ui";
import { useJobResult, type UnavailableResultState } from "../hooks/useJobResult";
import type { DisplayVerdict, ProductionResult, PublicEvidenceUnit } from "../types";

interface UnavailablePresentation {
  eyebrow: string;
  title: string;
  actionLabel: string;
  actionPath: (jobId: string | undefined) => string;
}

const UNAVAILABLE_PRESENTATION: Record<
  UnavailableResultState,
  UnavailablePresentation
> = {
  not_completed: {
    eyebrow: "Verification in progress",
    title: "Result not available yet",
    actionLabel: "Return to job status",
    actionPath: (jobId) => (jobId ? `/jobs/${encodeURIComponent(jobId)}` : "/"),
  },
  failed: {
    eyebrow: "Verification failed",
    title: "No successful result is available",
    actionLabel: "Review job status",
    actionPath: (jobId) => (jobId ? `/jobs/${encodeURIComponent(jobId)}` : "/"),
  },
  expired: {
    eyebrow: "Expired result",
    title: "This result has expired",
    actionLabel: "Start a new verification",
    actionPath: () => "/#verify",
  },
  unknown: {
    eyebrow: "Unknown result",
    title: "Result not found",
    actionLabel: "Start a new verification",
    actionPath: () => "/#verify",
  },
};

function verdictClassName(verdict: DisplayVerdict): string {
  return `result-verdict--${verdict.toLowerCase()}`;
}

function ResultOverview({ result }: { result: ProductionResult }) {
  const evidenceLabel =
    result.verdict.evidence_status === "sufficient"
      ? "Sufficient evidence"
      : "Insufficient evidence";

  return (
    <div className="result-overview">
      <Card
        aria-label="Authoritative claim"
        className="result-claim"
        role="region"
        variant="glass"
      >
        <p className="result-panel-eyebrow">Authoritative claim</p>
        <blockquote>{result.claim}</blockquote>
      </Card>

      <Card
        aria-label="Final verdict"
        className={`result-verdict ${verdictClassName(result.verdict.display_verdict)}`}
        role="region"
        variant="raised"
      >
        <div>
          <p className="result-panel-eyebrow">Final verdict</p>
          <p className="result-verdict__value">
            {result.verdict.display_verdict.toUpperCase()}
          </p>
        </div>
        <Badge
          tone={
            result.verdict.evidence_status === "sufficient" ? "success" : "warning"
          }
          withDot
        >
          {evidenceLabel}
        </Badge>
        <p className="result-verdict__boundary">
          Displayed directly from the authoritative production outcome.
        </p>
      </Card>
    </div>
  );
}

function ResultConfidence({ result }: { result: ProductionResult }) {
  const probabilities = Object.entries(result.verdict.probabilities);

  return (
    <Card
      aria-label="Authoritative model probabilities"
      className="result-confidence"
      role="region"
      variant="glass"
    >
      <div className="result-section-heading result-section-heading--split">
        <div>
          <p>Confidence view</p>
          <h2>Authoritative model probabilities</h2>
        </div>
        <Badge tone={probabilities.length > 0 ? "info" : "neutral"}>
          {probabilities.length > 0 ? "Backend provided" : "Not available"}
        </Badge>
      </div>

      {probabilities.length > 0 ? (
        <div className="result-confidence__list">
          {probabilities.map(([label, value]) => {
            const percentage = Math.max(0, Math.min(1, value)) * 100;
            return (
              <div className="result-confidence__item" key={label}>
                <div>
                  <span>{label}</span>
                  <strong>{percentage.toFixed(1)}%</strong>
                </div>
                <div
                  aria-label={`${label} probability ${percentage.toFixed(1)} percent`}
                  aria-valuemax={100}
                  aria-valuemin={0}
                  aria-valuenow={Number(percentage.toFixed(1))}
                  className="result-confidence__track"
                  role="progressbar"
                >
                  <span style={{ width: `${percentage}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="result-confidence__empty">
          No probability distribution was returned in the public result.
        </p>
      )}
      <p className="result-confidence__boundary">
        Values are visualized from the public response and do not determine or
        recompute the displayed verdict.
      </p>
    </Card>
  );
}

function ResultMetadata({
  jobId,
  result,
}: {
  jobId: string;
  result: ProductionResult;
}) {
  const items = [
    { label: "Job ID", value: jobId },
    { label: "Session ID", value: result.session_id },
    { label: "Evidence status", value: result.verdict.evidence_status },
    { label: "Runtime", value: `${String(result.runtime_ms)} ms` },
    { label: "Sufficiency reason", value: result.sufficiency.reason_code },
    { label: "Candidate units", value: String(result.sufficiency.g1_exposure_count) },
    { label: "Selected units", value: String(result.sufficiency.top_k_count) },
  ];

  return (
    <Card className="result-metadata">
      <div className="result-section-heading">
        <div>
          <p>Public result metadata</p>
          <h2>Verification record</h2>
        </div>
      </div>
      <dl>
        {items.map((item) => (
          <div key={item.label}>
            <dt>{item.label}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}

function MissingSelectedUnit({
  topKDisplayRank,
  unitId,
}: {
  topKDisplayRank: number;
  unitId: string;
}) {
  return (
    <Card className="evidence-unit evidence-unit--selected evidence-unit--missing">
      <div className="evidence-unit__topline">
        <Badge tone="success">Frozen G1 selected</Badge>
        <span
          aria-label={`Top-k explanation display rank ${String(topKDisplayRank)}`}
          className="evidence-unit__top-k-rank"
        >
          #{topKDisplayRank}
        </span>
      </div>
      <div className="evidence-unit__identity">
        <p>Unit identifier</p>
        <code>{unitId}</code>
      </div>
      <p>
        Detailed unit content was not included in the public candidate-unit response.
      </p>
    </Card>
  );
}

function EvidenceHierarchy({ result }: { result: ProductionResult }) {
  const candidates = result.evidence.g1_exposure_units;
  const selectedIds = result.evidence.g1_top_k_explanation_unit_ids;
  const presentation = deriveSelectionPresentation(candidates, selectedIds);
  const candidateById = new Map(
    candidates.map(
      (unit, index): [
        string,
        {
          unit: PublicEvidenceUnit;
          selectionPresentation: (typeof presentation)[number];
        },
      ] => [
        unit.unit_id,
        { unit, selectionPresentation: presentation[index] },
      ],
    ),
  );

  return (
    <div className="result-evidence">
      <section
        aria-labelledby="selected-units-heading"
        className="result-flow-stage result-flow-stage--selected"
      >
        <div className="result-flow-stage__heading">
          <div>
            <p>Explanation selection</p>
            <h2 id="selected-units-heading">Frozen G1 Top-k Selected Units</h2>
            <span>
              These units are returned by the frozen claim–unit selection head for
              explanation.
            </span>
          </div>
          <Badge tone="success">{selectedIds.length} selected</Badge>
        </div>

        <Card className="result-scientific-boundary" variant="subtle">
          <strong>
            Top-k selection is explanation-only. The final Frozen G1 verdict is not
            computed from these units alone.
          </strong>
          <p>
            The final verdict is composed from veracity logits over all valid Frozen G1
            candidate units using the frozen class-wise max-pooling rule.
          </p>
        </Card>

        {selectedIds.length > 0 ? (
          <div className="evidence-unit-grid evidence-unit-grid--selected">
            {selectedIds.map((unitId, selectedIndex) => {
              const candidate = candidateById.get(unitId);
              return candidate ? (
                <EvidenceUnitCard
                  key={unitId}
                  selectionPresentation={candidate.selectionPresentation}
                  unit={candidate.unit}
                  variant="selected"
                />
              ) : (
                <MissingSelectedUnit
                  key={unitId}
                  topKDisplayRank={selectedIndex + 1}
                  unitId={unitId}
                />
              );
            })}
          </div>
        ) : (
          <EmptyState
            description="No explanation-unit identifiers were returned by the authoritative result."
            eyebrow="Explanation selection"
            headingLevel={3}
            title="No explanation available"
          />
        )}
      </section>

      <section
        aria-labelledby="selection-method-heading"
        className="result-selection-method"
      >
        <div className="result-section-heading">
          <div>
            <p>Scientific interpretation</p>
            <h2 id="selection-method-heading">How these units were selected</h2>
          </div>
        </div>

        <div className="result-selection-branches">
          <Card className="result-selection-branch" variant="glass">
            <Badge tone="info">Explanation branch</Badge>
            <ol aria-label="Frozen G1 explanation selection process">
              <li>Claim + ordered candidate pool</li>
              <li>Claim–unit scoring</li>
              <li>Raw selection ranking scores</li>
              <li>Top-k explanation selection</li>
            </ol>
          </Card>
          <Card className="result-selection-branch" variant="glass">
            <Badge tone="research">Prediction branch</Badge>
            <ol aria-label="Frozen G1 final verdict composition process">
              <li>All valid unit veracity logits</li>
              <li>Class-wise max pooling</li>
              <li>Final verdict</li>
            </ol>
          </Card>
        </div>

        <Card className="result-score-explanation" variant="subtle">
          <p>
            Selection and prediction are separate branches. Selection scores rank
            units for explanation, while unit-level veracity logits from all valid
            Frozen G1 units contribute to the final class-wise max-pooled prediction.
          </p>
          <p>
            Selection scores are raw claim-conditioned ranking values, not
            probabilities. They are interpreted comparatively within the exposed
            candidate set.
          </p>
          <p>
            A negative score does not by itself mean that a unit is invalid or
            incorrect.
          </p>
        </Card>
      </section>

      <section aria-labelledby="candidate-units-heading" className="result-flow-stage">
        <div className="result-flow-stage__heading">
          <div>
            <p>Candidate analysis</p>
            <h2 id="candidate-units-heading">Full Frozen G1 Candidate Pool</h2>
            <span>
              The ordered candidate pool exposed to Frozen G1. Candidate order is
              preserved for reproducibility.
            </span>
          </div>
          <Badge tone="neutral">{candidates.length} returned</Badge>
        </div>
        {candidates.length > 0 ? (
          <div className="evidence-unit-grid">
            {candidates.map((unit, index) => (
              <EvidenceUnitCard
                key={unit.unit_id}
                selectionPresentation={presentation[index]}
                unit={unit}
              />
            ))}
          </div>
        ) : (
          <EmptyState
            description="The authoritative result contains no frozen-G1 candidate units."
            eyebrow="Candidate analysis"
            headingLevel={3}
            title="No candidate evidence available"
          />
        )}
      </section>
    </div>
  );
}

function SupplementalEvidence({
  jobId,
  onVisualXAIReady,
  result,
}: {
  jobId: string;
  onVisualXAIReady: () => void;
  result: ProductionResult;
}) {
  const units = result.evidence.visual_supplemental_units;

  return (
    <section aria-labelledby="supplemental-heading" className="result-supplemental">
      <div className="result-section-heading result-section-heading--split">
        <div>
          <p>Supplemental visual grounding</p>
          <h2 id="supplemental-heading">Supplemental Visual Observations</h2>
        </div>
        <p>
          Supplemental visual observations provide additional visual grounding and
          are not part of the Frozen G1 Text+OCR candidate pool.
        </p>
      </div>
      {units.length > 0 ? (
        <div className="evidence-unit-grid">
          {units.map((unit) => (
            <EvidenceUnitCard
              jobId={jobId}
              key={unit.unit_id}
              onVisualXAIReady={onVisualXAIReady}
              unit={unit}
              variant="supplemental"
            />
          ))}
        </div>
      ) : (
        <EmptyState
          description="The authoritative result contains no supplemental visual observations."
          eyebrow="Supplemental observations"
          headingLevel={3}
          title="No supplemental observations available"
        />
      )}
    </section>
  );
}

export function ResultPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { error, jobResult, loading, refresh, retry, unavailable } = useJobResult(jobId);
  const displayedJobId = jobResult?.jobId ?? jobId ?? "Unavailable";
  const unavailablePresentation = unavailable
    ? UNAVAILABLE_PRESENTATION[unavailable.state]
    : undefined;

  return (
    <section aria-labelledby="result-heading" className="result-page">
      <div className="result-page__header">
        <SectionHeader
          description="Review the authoritative verdict and its public Minimal Deceptive Unit evidence hierarchy."
          eyebrow="Explainable verification"
          headingId="result-heading"
          headingLevel={1}
          title="Verification Result"
        />
        <Link
          className="result-page__back-link"
          to={jobId ? `/jobs/${encodeURIComponent(jobId)}` : "/"}
        >
          Back to job status
        </Link>
      </div>

      <Card className="result-session-bar" variant="glass">
        <div>
          <p>Public job identifier</p>
          <code>{displayedJobId}</code>
        </div>
        <Badge
          tone={jobResult ? "success" : "neutral"}
          withDot={Boolean(jobResult)}
        >
          {jobResult ? "Authoritative result loaded" : "Result request"}
        </Badge>
      </Card>

      {loading ? (
        <LoadingState
          detail="Requesting the completed public outcome from the production API."
          label="Loading verification result"
        />
      ) : null}

      {unavailable && unavailablePresentation ? (
        <div className="result-unavailable">
          <EmptyState
            action={
              <Link
                className="result-page__back-link"
                to={unavailablePresentation.actionPath(jobId)}
              >
                {unavailablePresentation.actionLabel}
              </Link>
            }
            description={unavailable.message}
            eyebrow={unavailablePresentation.eyebrow}
            title={unavailablePresentation.title}
          />
          {unavailable.requestId ? (
            <p className="result-request-id">Request ID: {unavailable.requestId}</p>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <div className="result-error">
          <ErrorMessage
            message={error.message}
            requestId={error.requestId}
            title="Result temporarily unavailable"
          />
          <Button onClick={retry} variant="secondary">
            Retry result
          </Button>
        </div>
      ) : null}

      {jobResult ? (
        <>
          <ResultOverview result={jobResult.result} />
          <EvidenceHierarchy result={jobResult.result} />
          <SupplementalEvidence
            jobId={jobResult.jobId}
            onVisualXAIReady={refresh}
            result={jobResult.result}
          />
          <ResultConfidence result={jobResult.result} />
          <ResultMetadata jobId={jobResult.jobId} result={jobResult.result} />
        </>
      ) : null}
    </section>
  );
}
