import { useParams } from "react-router-dom";

export function JobStatusPage() {
  const { jobId } = useParams<{ jobId: string }>();

  return (
    <section aria-labelledby="job-heading" className="page-panel">
      <p className="page-eyebrow">Job status</p>
      <h1 id="job-heading">Verification job</h1>
      <p>Job identifier: {jobId ?? "Unavailable"}</p>
      <p>Live polling and workflow presentation will be added in a later phase.</p>
    </section>
  );
}

