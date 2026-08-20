import { useParams } from "react-router-dom";

export function ResultPage() {
  const { jobId } = useParams<{ jobId: string }>();

  return (
    <section aria-labelledby="result-heading" className="page-panel">
      <p className="page-eyebrow">Result</p>
      <h1 id="result-heading">Verification result</h1>
      <p>Job identifier: {jobId ?? "Unavailable"}</p>
      <p>Authoritative result rendering will be added in a later phase.</p>
    </section>
  );
}

