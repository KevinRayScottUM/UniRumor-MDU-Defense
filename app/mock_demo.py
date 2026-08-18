"""Run the deterministic fixture through the runtime skeleton."""

import json
from pathlib import Path

from pipeline import RuntimeConfig, RuntimeOrchestrator
from schemas import VerificationRequest


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    config = RuntimeConfig.from_yaml(REPOSITORY_ROOT / "configs" / "runtime_mock.yaml")
    request_data = json.loads(
        (REPOSITORY_ROOT / "tests" / "fixtures" / "mock_request.json").read_text(encoding="utf-8")
    )
    orchestrator = RuntimeOrchestrator(config)
    result = orchestrator.run(VerificationRequest.from_dict(request_data))
    print(f"session_id={result.session_id}")
    print(f"model_verdict={result.model_verdict.value}")
    print(f"display_verdict={result.display_verdict.value}")
    print(f"result_path={orchestrator.last_result_path}")
    print("warning=MOCK_NON_SCIENTIFIC_OUTPUT")


if __name__ == "__main__":
    main()
