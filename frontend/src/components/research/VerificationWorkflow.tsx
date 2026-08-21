const workflowStages = [
  {
    index: "01",
    label: "Define",
    title: "Claim + source video",
    description:
      "The exact focal claim and one video enter through the validated public request boundary.",
  },
  {
    index: "02",
    label: "Decompose",
    title: "Candidate evidence units",
    description:
      "Text, transcript, and OCR units form the eligible evidence view; visual observations remain supplemental.",
  },
  {
    index: "03",
    label: "Verify",
    title: "Frozen decision contract",
    description:
      "The production runtime owns evaluation and returns Fake, Real, or an NEI abstention through one closed boundary.",
  },
  {
    index: "04",
    label: "Explain",
    title: "Auditable public result",
    description:
      "The interface presents the authoritative verdict, candidate units, and ordered explanation references.",
  },
] as const;

export function VerificationWorkflow() {
  return (
    <ol aria-label="UniRumor-MDU verification workflow" className="research-workflow">
      {workflowStages.map((stage) => (
        <li className="research-workflow__stage" key={stage.index}>
          <div className="research-workflow__index">
            <span>{stage.index}</span>
          </div>
          <div className="research-workflow__copy">
            <p>{stage.label}</p>
            <h3>{stage.title}</h3>
            <span>{stage.description}</span>
          </div>
        </li>
      ))}
    </ol>
  );
}
