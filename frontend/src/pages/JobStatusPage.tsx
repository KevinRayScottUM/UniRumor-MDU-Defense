import { Link, useParams } from "react-router-dom";

import { useJobStatusPolling } from "../hooks/useJobStatusPolling";
import type { JobState, JobStatus } from "../types";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorMessage,
  LoadingState,
  SectionHeader,
  type BadgeTone,
} from "../components/ui";

type MonitorState = JobState | "unknown";
type StepState = "complete" | "active" | "pending" | "failed" | "unavailable";

interface StatePresentation {
  label: string;
  tone: BadgeTone;
  summary: string;
}

interface TimelineStep {
  index: string;
  title: string;
  description: string;
  state: StepState;
}

const STATE_PRESENTATION: Record<MonitorState, StatePresentation> = {
  accepted: {
    label: "Accepted",
    tone: "info",
    summary: "The production API has accepted this verification session.",
  },
  queued: {
    label: "Queued",
    tone: "info",
    summary: "The session is waiting in the server-managed execution queue.",
  },
  running: {
    label: "Running",
    tone: "research",
    summary: "The backend reports that verification execution is active.",
  },
  completed: {
    label: "Completed",
    tone: "success",
    summary: "The backend reports a completed session with a result available.",
  },
  failed: {
    label: "Failed",
    tone: "danger",
    summary: "The backend reports that verification execution failed.",
  },
  expired: {
    label: "Expired",
    tone: "warning",
    summary: "The retained public job record has expired.",
  },
  unknown: {
    label: "Unknown",
    tone: "neutral",
    summary: "No authoritative status is currently available for this job ID.",
  },
};

const STEP_LABELS: Record<StepState, string> = {
  complete: "Complete",
  active: "Current",
  pending: "Waiting",
  failed: "Failed",
  unavailable: "Unavailable",
};

function statePresentation(state: string): StatePresentation {
  return STATE_PRESENTATION[state as MonitorState] ?? STATE_PRESENTATION.unknown;
}

function isTerminal(state: JobState): boolean {
  return state === "completed" || state === "failed" || state === "expired";
}

function timelineFor(job: JobStatus): TimelineStep[] {
  const executionState: StepState =
    job.state === "running"
      ? "active"
      : job.state === "completed"
        ? "complete"
        : job.state === "failed"
          ? "failed"
          : job.state === "expired"
            ? "unavailable"
            : "pending";
  const resultState: StepState =
    job.state === "completed"
      ? "complete"
      : job.state === "failed" || job.state === "expired"
        ? "unavailable"
        : "pending";

  return [
    {
      index: "01",
      title: "Submission received",
      description: "Confirmed by the authoritative public job record.",
      state: "complete",
    },
    {
      index: "02",
      title: "Workspace prepared",
      description: "Established before the job entered production scheduling.",
      state: "complete",
    },
    {
      index: "03",
      title: "Verification execution",
      description:
        job.state === "running"
          ? "The backend reports active execution."
          : job.state === "completed"
            ? "Execution completed according to the backend."
            : job.state === "failed"
              ? "The backend reports an execution failure."
              : "Waiting for the backend to report active execution.",
      state: executionState,
    },
    {
      index: "04",
      title: "Result available",
      description:
        job.state === "completed"
          ? "The authoritative result endpoint is available."
          : job.state === "failed"
            ? "No successful result is available for this failed session."
            : "Available only after the backend reports completion.",
      state: resultState,
    },
  ];
}

function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(parsed);
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds} ms`;
  const seconds = milliseconds / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes} min ${remainingSeconds} s`;
}

function JobTimeline({ job }: { job: JobStatus }) {
  return (
    <Card className="job-timeline" variant="raised">
      <div className="job-panel-heading">
        <div>
          <p>Execution visualization</p>
          <h2>Verification workflow</h2>
        </div>
        {!isTerminal(job.state) ? (
          <span className="job-polling-indicator" role="status">
            <span aria-hidden="true" className="job-polling-indicator__dot" />
            Automatic updates active
          </span>
        ) : null}
      </div>
      <ol className="job-timeline__list">
        {timelineFor(job).map((step) => (
          <li
            className={`job-timeline__step job-timeline__step--${step.state}`}
            key={step.index}
          >
            <div className="job-timeline__rail" aria-hidden="true">
              <span>{step.index}</span>
            </div>
            <div className="job-timeline__copy">
              <h3>{step.title}</h3>
              <p>{step.description}</p>
            </div>
            <Badge
              tone={
                step.state === "failed"
                  ? "danger"
                  : step.state === "complete"
                    ? "success"
                    : step.state === "active"
                      ? "research"
                      : "neutral"
              }
            >
              {STEP_LABELS[step.state]}
            </Badge>
          </li>
        ))}
      </ol>
    </Card>
  );
}

function SessionDetails({ job }: { job: JobStatus }) {
  const details = [
    { label: "Created", value: formatTimestamp(job.created_at) },
    job.queue_position !== null
      ? { label: "Queue position", value: String(job.queue_position) }
      : null,
    { label: "Queue elapsed", value: formatDuration(job.queue_elapsed_ms) },
    job.started_at
      ? { label: "Started", value: formatTimestamp(job.started_at) }
      : null,
    job.started_at
      ? {
          label: "Execution elapsed",
          value: formatDuration(job.execution_elapsed_ms),
        }
      : null,
    job.finished_at
      ? { label: "Finished", value: formatTimestamp(job.finished_at) }
      : null,
    job.expires_at
      ? { label: "Retained until", value: formatTimestamp(job.expires_at) }
      : null,
  ].filter((detail): detail is { label: string; value: string } => detail !== null);

  return (
    <Card className="job-session-details">
      <div className="job-panel-heading">
        <div>
          <p>Public metadata</p>
          <h2>Session details</h2>
        </div>
      </div>
      <dl>
        {details.map((detail) => (
          <div key={detail.label}>
            <dt>{detail.label}</dt>
            <dd>{detail.value}</dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}

export function JobStatusPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { job, loading, pollingError, retry, unavailable } =
    useJobStatusPolling(jobId);
  const currentState: MonitorState | undefined = unavailable?.state ?? job?.state;
  const presentation = statePresentation(currentState ?? "unknown");
  const displayedJobId = job?.job_id ?? jobId ?? "Unavailable";

  return (
    <section aria-labelledby="job-heading" className="job-monitor-page">
      <div className="job-monitor-page__header">
        <SectionHeader
          description="Follow the server-owned lifecycle of this verification request. Status changes appear only when returned by the public API."
          eyebrow="Verification Session"
          headingId="job-heading"
          headingLevel={1}
          title="Verification Session"
        />
        <Link className="job-monitor-page__new-link" to="/#verify">
          New verification
        </Link>
      </div>

      <Card className="job-session-summary" variant="glass">
        <div>
          <p className="job-session-summary__label">
            {job ? "Job ID" : "Requested job ID"}
          </p>
          <p className="technical-value">{displayedJobId}</p>
        </div>
        <div className="job-session-summary__status" aria-live="polite">
          <p className="job-session-summary__label">Current status</p>
          {loading && !job ? (
            <Badge tone="neutral">Checking public status</Badge>
          ) : (
            <Badge tone={presentation.tone} withDot>
              {presentation.label}
            </Badge>
          )}
        </div>
        <p className="job-session-summary__description">
          {loading && !job
            ? "Requesting the latest authoritative job record."
            : presentation.summary}
        </p>
      </Card>

      {loading && !job ? (
        <LoadingState
          detail="Requesting the latest state from the production API."
          label="Loading verification session"
        />
      ) : null}

      {unavailable ? (
        <div className="job-monitor-unavailable">
          <EmptyState
            action={
              <Link className="job-monitor-page__new-link" to="/#verify">
                Start a new verification
              </Link>
            }
            description={unavailable.message}
            eyebrow={unavailable.state === "expired" ? "Expired session" : "Unknown session"}
            title={unavailable.state === "expired" ? "This session has expired" : "Job not found"}
          />
          {unavailable.requestId ? (
            <p className="job-monitor-request-id">
              Request ID: {unavailable.requestId}
            </p>
          ) : null}
        </div>
      ) : null}

      {!loading && !job && !unavailable && pollingError ? (
        <div className="job-monitor-error">
          <ErrorMessage
            message={pollingError.message}
            requestId={pollingError.requestId}
            title="Status temporarily unavailable"
          />
          <Button onClick={retry} variant="secondary">
            Retry status
          </Button>
        </div>
      ) : null}

      {job ? (
        <>
          {pollingError ? (
            <div className="job-monitor-inline-error">
              <ErrorMessage
                message={pollingError.message}
                requestId={pollingError.requestId}
                title="Status update interrupted"
              />
              <Button onClick={retry} size="small" variant="secondary">
                Retry now
              </Button>
            </div>
          ) : null}

          <div className="job-monitor-grid">
            <JobTimeline job={job} />
            <SessionDetails job={job} />
          </div>

          {job.state === "failed" && job.failure ? (
            <Card className="job-failure-panel">
              <Badge tone="danger">Execution failure</Badge>
              <div>
                <h2>{job.failure.message}</h2>
                <p>
                  Incident ID: <span>{job.failure.incident_id}</span>
                </p>
              </div>
            </Card>
          ) : null}

          {job.state === "completed" ? (
            <Card className="job-result-available" variant="subtle">
              <div>
                <p>Authoritative result</p>
                <h2>The completed result is ready for review.</h2>
              </div>
              <Link
                className="job-monitor-page__result-link"
                to={`/jobs/${encodeURIComponent(job.job_id)}/result`}
              >
                View result
              </Link>
            </Card>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

