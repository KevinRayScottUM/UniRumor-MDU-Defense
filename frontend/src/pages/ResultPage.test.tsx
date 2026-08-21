import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClientError } from "../api";
import { AppRoutes } from "../app/App";
import { apiClient } from "../app/api";
import type {
  DisplayVerdict,
  JobResultResponse,
  ModelVerdict,
  PublicEvidenceUnit,
} from "../types";

const JOB_ID = "job_0123456789abcdef0123456789abcdef";

function makeUnit(
  unitId: string,
  overrides: Partial<PublicEvidenceUnit> = {},
): PublicEvidenceUnit {
  return {
    unit_id: unitId,
    source_type: "transcript",
    text: `Authoritative evidence text for ${unitId}.`,
    start_time: 12.5,
    end_time: 18,
    frame_id: null,
    bbox: null,
    confidence: null,
    producer: "public-transcript-adapter",
    eligible_for_frozen_g1: true,
    selection_score: 0.91,
    logits: { fake: 1.25, real: -0.4 },
    extraction_method: "timestamped_transcript_segment",
    source_index: 2,
    frame_ids: [],
    evidence_refs: ["segment-02"],
    source_unit_ids: [],
    observation_type: null,
    ...overrides,
  };
}

function makeResultResponse({
  candidates = [makeUnit("unit-transcript")],
  claim = "A public focal claim.",
  displayVerdict = "Fake",
  modelVerdict = "fake",
  selectedIds = ["unit-transcript"],
  supplemental = [],
}: {
  candidates?: PublicEvidenceUnit[];
  claim?: string;
  displayVerdict?: DisplayVerdict;
  modelVerdict?: ModelVerdict;
  selectedIds?: string[];
  supplemental?: PublicEvidenceUnit[];
} = {}): JobResultResponse {
  const modelWasRun = modelVerdict !== "not_run";
  const evidenceStatus = modelWasRun ? "sufficient" : "insufficient";
  return {
    api_version: "v1",
    job_id: JOB_ID,
    outcome: {
      schema_version: 1,
      status: "success",
      failure: null,
      result: {
        schema_version: 1,
        session_id: JOB_ID,
        claim,
        verdict: {
          model_verdict: modelVerdict,
          display_verdict: displayVerdict,
          evidence_status: evidenceStatus,
          sample_logits: modelWasRun ? { fake: 1.25, real: -0.4 } : {},
          probabilities: modelWasRun ? { fake: 0.84, real: 0.16 } : {},
          class_winners: modelWasRun ? { fake: "unit-transcript" } : {},
          checkpoint_sha256: modelWasRun ? "checkpoint-sha" : null,
        },
        sufficiency: {
          status: evidenceStatus,
          reason_code: modelWasRun
            ? "frozen_g1_evidence_available_and_model_completed"
            : "no_frozen_g1_eligible_evidence",
          model_was_run: modelWasRun,
          g1_exposure_count: candidates.length,
          transcript_exposure_count: candidates.filter(
            (unit) => unit.source_type === "transcript",
          ).length,
          ocr_exposure_count: candidates.filter((unit) => unit.source_type === "ocr")
            .length,
          visual_unit_count: supplemental.length,
          top_k_count: selectedIds.length,
          supplemental_visual_present: supplemental.length > 0,
        },
        evidence: {
          g1_exposure_units: candidates,
          g1_top_k_explanation_unit_ids: selectedIds,
          visual_supplemental_units: supplemental,
        },
        runtime_ms: 123.456,
      },
    },
  };
}

function renderResult() {
  return render(
    <MemoryRouter initialEntries={[`/jobs/${JOB_ID}/result`]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("explainable result experience", () => {
  it("renders the authoritative verdict, long claim, evidence order, and metadata", async () => {
    const longClaim =
      "This deliberately long authoritative claim checks that the research result layout preserves backend text without truncation, reconstruction, or scientific reinterpretation across the summary and evidence hierarchy.";
    const transcript = makeUnit("unit-transcript");
    const ocr = makeUnit("unit-ocr", {
      source_type: "ocr",
      text: "OCR evidence returned by the backend.",
      start_time: null,
      end_time: null,
      frame_id: "frame_004",
      bbox: [1, 2, 30, 40],
      producer: "public-ocr-adapter",
      extraction_method: "frame_ocr",
      selection_score: 0.72,
      evidence_refs: ["frame_004"],
    });
    const visual = makeUnit("unit-visual", {
      source_type: "visual_observation",
      text: "A visible banner appears in the selected frame.",
      start_time: null,
      end_time: null,
      frame_id: "frame_008",
      frame_ids: ["frame_008"],
      producer: "public-visual-observer",
      eligible_for_frozen_g1: false,
      selection_score: null,
      logits: null,
      extraction_method: "claim_blind_visual_observation",
      source_index: null,
      evidence_refs: ["frame_008"],
      observation_type: "visible_content",
    });
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      makeResultResponse({
        candidates: [transcript, ocr],
        claim: longClaim,
        selectedIds: ["unit-ocr", "unit-transcript"],
        supplemental: [visual],
      }),
    );

    renderResult();

    expect(await screen.findByText("FAKE")).toBeVisible();
    expect(screen.getAllByText(longClaim)).toHaveLength(2);
    expect(screen.getByRole("heading", { name: "Candidate units" })).toBeVisible();
    const selectedSection = screen
      .getByRole("heading", { name: "Selected explanation units" })
      .closest("section");
    expect(selectedSection).not.toBeNull();
    const selectedUnitIds = within(selectedSection as HTMLElement)
      .getAllByText(/^unit-/)
      .map((node) => node.textContent);
    expect(selectedUnitIds).toEqual(["unit-ocr", "unit-transcript"]);
    expect(screen.getAllByText("Selection score 0.72")).toHaveLength(2);
    expect(screen.getAllByText("frame_004")).toHaveLength(4);
    expect(
      screen.getByRole("heading", { name: "Supplemental observations" }),
    ).toBeVisible();
    expect(screen.getByText("Supplemental")).toBeVisible();
    expect(
      screen.getByRole("region", { name: "Authoritative claim" }),
    ).toBeVisible();
    expect(screen.getByRole("region", { name: "Final verdict" })).toBeVisible();
  });

  it("renders authoritative NEI with explicit empty explanation states", async () => {
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      makeResultResponse({
        candidates: [],
        displayVerdict: "NEI",
        modelVerdict: "not_run",
        selectedIds: [],
      }),
    );

    renderResult();

    expect(await screen.findByText("NEI")).toBeVisible();
    expect(screen.getByText("Insufficient evidence")).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "No candidate evidence available" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "No explanation available" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", {
        name: "No supplemental observations available",
      }),
    ).toBeVisible();
  });

  it("renders the authoritative REAL verdict without recomputation", async () => {
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      makeResultResponse({
        displayVerdict: "Real",
        modelVerdict: "real",
      }),
    );

    renderResult();

    expect(await screen.findByText("REAL")).toBeVisible();
    expect(screen.queryByText("FAKE")).not.toBeInTheDocument();
    expect(screen.queryByText("NEI")).not.toBeInTheDocument();
  });

  it("shows missing optional unit metadata without inventing values", async () => {
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      makeResultResponse({
        candidates: [
          makeUnit("unit-minimal", {
            start_time: null,
            end_time: null,
            frame_id: null,
            bbox: null,
            confidence: null,
            selection_score: null,
            source_index: null,
            frame_ids: [],
            evidence_refs: [],
            source_unit_ids: [],
            observation_type: null,
          }),
        ],
        selectedIds: [],
      }),
    );

    renderResult();

    expect(
      await screen.findByText("No additional public metadata was provided for this unit."),
    ).toBeVisible();
    expect(screen.queryByText(/Selection score/)).not.toBeInTheDocument();
  });

  it("maps failed, expired, and not-completed public errors", async () => {
    const getResult = vi.spyOn(apiClient, "getJobResult").mockRejectedValueOnce(
      new ApiClientError(409, "job_failed", "Job execution failed.", "req_failed"),
    );
    const failed = renderResult();

    expect(
      await screen.findByRole("heading", {
        name: "No successful result is available",
      }),
    ).toBeVisible();
    expect(screen.getByText("Job execution failed.")).toBeVisible();
    expect(screen.getByText("Request ID: req_failed")).toBeVisible();
    failed.unmount();

    getResult.mockRejectedValueOnce(
      new ApiClientError(410, "job_expired", "Job has expired.", "req_expired"),
    );
    const expired = renderResult();
    expect(
      await screen.findByRole("heading", { name: "This result has expired" }),
    ).toBeVisible();
    expired.unmount();

    getResult.mockRejectedValueOnce(
      new ApiClientError(
        409,
        "job_not_completed",
        "Job has not completed.",
        "req_active",
      ),
    );
    renderResult();
    expect(
      await screen.findByRole("heading", { name: "Result not available yet" }),
    ).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Return to job status" }),
    ).toHaveAttribute("href", `/jobs/${JOB_ID}`);
  });

  it("recovers from a network failure through the explicit retry", async () => {
    const getResult = vi
      .spyOn(apiClient, "getJobResult")
      .mockRejectedValueOnce(new TypeError("network unavailable"))
      .mockResolvedValueOnce(makeResultResponse());
    renderResult();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The verification result could not be retrieved.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry result" }));

    expect(await screen.findByText("FAKE")).toBeVisible();
    await waitFor(() => expect(getResult).toHaveBeenCalledTimes(2));
  });
});
