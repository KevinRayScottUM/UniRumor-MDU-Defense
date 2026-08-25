import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClientError } from "../api";
import { AppRoutes } from "../app/App";
import { apiClient } from "../app/api";
import {
  VISUAL_XAI_QA_FRAME,
  VISUAL_XAI_QA_OBSERVATION,
  VISUAL_XAI_QA_ORIGINAL,
  VISUAL_XAI_QA_PHRASE_HEATMAP,
  VISUAL_XAI_QA_WHOLE_HEATMAP,
} from "../test/visualXaiFixture";
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
    expect(screen.getByText(longClaim)).toBeVisible();
    const selectedHeading = screen.getByRole("heading", {
      name: "Frozen G1 Top-k Selected Units",
    });
    const candidateHeading = screen.getByRole("heading", {
      name: "Full Frozen G1 Candidate Pool",
    });
    expect(
      selectedHeading.compareDocumentPosition(candidateHeading) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    const selectedSection = screen
      .getByRole("heading", { name: "Frozen G1 Top-k Selected Units" })
      .closest("section");
    expect(selectedSection).not.toBeNull();
    const selectedUnitIds = within(selectedSection as HTMLElement)
      .getAllByText(/^unit-/)
      .map((node) => node.textContent);
    expect(selectedUnitIds).toEqual(["unit-ocr", "unit-transcript"]);
    expect(screen.getAllByText("+0.7200")).toHaveLength(2);
    expect(screen.getAllByText("frame_004")).toHaveLength(4);
    expect(
      screen.getByRole("heading", { name: "Supplemental Visual Observations" }),
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
    const wrappedBackendResponse = makeResultResponse({
      displayVerdict: "Real",
      modelVerdict: "real",
    });
    expect(
      wrappedBackendResponse.outcome.result.verdict.display_verdict,
    ).toBe("Real");
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      wrappedBackendResponse,
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
    const candidateSection = screen
      .getByRole("heading", { name: "Full Frozen G1 Candidate Pool" })
      .closest("section");
    expect(candidateSection).not.toBeNull();
    expect(
      within(candidateSection as HTMLElement).getByText(
        "Raw selection ranking score",
      ),
    ).toBeVisible();
    expect(
      within(candidateSection as HTMLElement).getAllByText("Not available"),
    ).toHaveLength(2);
  });

  it("presents authoritative Top-k first with scientific ranking and OCR semantics", async () => {
    const candidates = [
      makeUnit("ocr_0001", {
        source_type: "ocr",
        confidence: 0.952214777469635,
        selection_score: -0.21643970906734467,
      }),
      makeUnit("transcript_exposure_0005", { selection_score: 0.44004 }),
      makeUnit("transcript_exposure_0000", { selection_score: 0.198685 }),
      makeUnit("transcript_exposure_0010", { selection_score: 0.122902 }),
      makeUnit("transcript_exposure_0001", { selection_score: 0.11554 }),
      makeUnit("transcript_exposure_0008", { selection_score: -0.019889 }),
      makeUnit("unit-06", { selection_score: -0.03 }),
      makeUnit("unit-07", { selection_score: -0.05 }),
      makeUnit("unit-08", { selection_score: -0.08 }),
      makeUnit("unit-09", { selection_score: -0.1 }),
      makeUnit("unit-10", { selection_score: -0.15 }),
      makeUnit("unit-11", { selection_score: -0.19 }),
      makeUnit("unit-12", { selection_score: -0.2 }),
      makeUnit("unit-14", { selection_score: -0.3 }),
      makeUnit("unit-15", { selection_score: -0.4 }),
      makeUnit("unit-16", { selection_score: -0.5 }),
      makeUnit("unit-17", { selection_score: -0.6 }),
      makeUnit("unit-18", { selection_score: null }),
    ];
    const selectedIds = [
      "transcript_exposure_0005",
      "transcript_exposure_0000",
      "transcript_exposure_0010",
      "transcript_exposure_0001",
      "transcript_exposure_0008",
    ];
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      makeResultResponse({ candidates, selectedIds }),
    );

    renderResult();

    const selectedSection = (await screen.findByRole("heading", {
      name: "Frozen G1 Top-k Selected Units",
    })).closest("section") as HTMLElement;
    const candidateSection = screen
      .getByRole("heading", { name: "Full Frozen G1 Candidate Pool" })
      .closest("section") as HTMLElement;

    expect(
      within(selectedSection)
        .getAllByText(/^transcript_exposure_/)
        .map((node) => node.textContent),
    ).toEqual(selectedIds);
    expect(
      within(candidateSection)
        .getAllByText(/^(ocr_0001|transcript_exposure_|unit-)/)
        .map((node) => node.textContent),
    ).toEqual(candidates.map((unit) => unit.unit_id));

    const ocrIdentifier = within(candidateSection).getByText("ocr_0001");
    const ocrCard = ocrIdentifier.closest(".evidence-unit");
    expect(ocrCard).not.toBeNull();
    expect(within(ocrCard as HTMLElement).getByText("-0.2164")).toBeVisible();
    expect(within(ocrCard as HTMLElement).getByText("13 / 18")).toBeVisible();
    expect(within(ocrCard as HTMLElement).getByText("Not selected")).toBeVisible();
    expect(
      within(ocrCard as HTMLElement).getByText("OCR recognition confidence"),
    ).toBeVisible();
    expect(within(ocrCard as HTMLElement).getByText("95.2%")).toBeVisible();

    expect(
      screen.getByText(/Selection scores are raw claim-conditioned ranking values, not probabilities/),
    ).toBeVisible();
    expect(
      screen.getByText(/A negative score does not by itself mean that a unit is invalid or incorrect/),
    ).toBeVisible();
    expect(
      screen.getByText(/Top-k selection is explanation-only/),
    ).toBeVisible();
    expect(
      screen.getByText(/all valid Frozen G1 candidate units using the frozen class-wise max-pooling rule/),
    ).toBeVisible();
    expect(screen.queryByText(/selection probability/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Top-k determines the verdict/i)).not.toBeInTheDocument();
    expect(
      screen.queryByText(/selected units are the only basis/i),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(/negative score means invalid/i),
    ).not.toBeInTheDocument();
  });

  it("renders real OCR frames, recorded regions, and the accessible lightbox", async () => {
    const image =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";
    const ocr = makeUnit("unit-ocr-grounded", {
      source_type: "ocr",
      text: "CPAC 2018",
      frame_id: "frame_004",
      extraction_method: "frame_ocr",
      evidence_frames: [
        {
          frame_id: "frame_004",
          frame_index: 4,
          timestamp: 12.5,
          original_image: image,
          annotated_image: null,
          bbox: [10, 8, 120, 42],
          regions: [
            { text: "CPAC", bbox: [10, 8, 62, 42], confidence: 0.97 },
            { text: "2018", bbox: [68, 8, 120, 42], confidence: 0.93 },
          ],
          explanation: "OCR text is grounded in 2 recorded regions on this frame.",
        },
      ],
    });
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      makeResultResponse({ candidates: [ocr], selectedIds: [] }),
    );

    renderResult();

    expect(await screen.findByText("OCR Evidence Frames")).toBeVisible();
    expect(screen.getByText(/grounded in 2 recorded OCR regions/)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Inspect frame_004" }));

    const dialog = screen.getByRole("dialog", { name: "frame_004" });
    expect(within(dialog).getByText("Original public evidence frame")).toBeVisible();
    expect(within(dialog).getByText("Annotated OCR regions")).toBeVisible();
    expect(within(dialog).getByText("CPAC")).toBeVisible();
    expect(within(dialog).getByText("2018")).toBeVisible();

    const annotated = within(dialog).getByAltText("frame_004 annotated evidence");
    Object.defineProperty(annotated, "naturalWidth", { configurable: true, value: 160 });
    Object.defineProperty(annotated, "naturalHeight", { configurable: true, value: 90 });
    fireEvent.load(annotated);
    expect(
      within(dialog).getByRole("img", { name: "2 grounded OCR regions" }),
    ).toBeVisible();

  });

  it("renders faithful visual XAI, switches phrase maps, and preserves lightbox access", async () => {
    const visual = makeUnit("visual-xai-1", {
      source_type: "visual_observation",
      text: VISUAL_XAI_QA_OBSERVATION,
      frame_id: "F001",
      frame_ids: ["F001"],
      evidence_refs: ["F001"],
      producer: "Qwen/Qwen2.5-VL-7B-Instruct",
      eligible_for_frozen_g1: false,
      selection_score: null,
      logits: null,
      observation_type: "scene",
      evidence_frames: [VISUAL_XAI_QA_FRAME],
    });
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      makeResultResponse({ supplemental: [visual] }),
    );

    renderResult();

    const viewer = await screen.findByRole("region", {
      name: "XAI attribution for F001",
    });
    expect(within(viewer).getByText("Qwen occlusion attribution")).toBeVisible();
    expect(within(viewer).getByText("Occlusion attribution")).toBeVisible();
    expect(within(viewer).getByText(/does not affect the authoritative verification verdict/)).toBeVisible();
    expect(within(viewer).getByText(/does not participate in the Frozen G1 verdict/)).toBeVisible();
    expect(within(viewer).getByAltText("F001 XAI attribution")).toHaveAttribute(
      "src",
      VISUAL_XAI_QA_WHOLE_HEATMAP,
    );

    fireEvent.click(within(viewer).getByRole("button", { name: "Microphones" }));
    expect(within(viewer).getByAltText("F001 XAI attribution")).toHaveAttribute(
      "src",
      VISUAL_XAI_QA_PHRASE_HEATMAP,
    );
    expect(within(viewer).getByText(/support for the phrase “Microphones”/)).toBeVisible();

    const thumbnail = screen.getByRole("button", { name: "Inspect F001" });
    thumbnail.focus();
    fireEvent.click(thumbnail);
    let dialog = screen.getByRole("dialog", { name: "F001" });
    fireEvent.click(within(dialog).getByRole("button", { name: "Original" }));
    expect(within(dialog).getByAltText("F001 original")).toHaveAttribute("src", VISUAL_XAI_QA_ORIGINAL);
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "F001" })).not.toBeInTheDocument());
    expect(thumbnail).toHaveFocus();

    fireEvent.click(thumbnail);
    dialog = screen.getByRole("dialog", { name: "F001" });
    fireEvent.click(within(dialog).getByRole("button", { name: "Close evidence viewer" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "F001" })).not.toBeInTheDocument());

    fireEvent.click(thumbnail);
    dialog = screen.getByRole("dialog", { name: "F001" });
    fireEvent.click(dialog);
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "F001" })).not.toBeInTheDocument());
  });

  it("loads the verdict first and lazily requests visual XAI without blocking the result", async () => {
    const pendingFrame = {
      ...VISUAL_XAI_QA_FRAME,
      xai: {
        ...VISUAL_XAI_QA_FRAME.xai!,
        status: "not_requested" as const,
        attribution_maps: [],
        cache_hit: false,
        queue_wait_ms: null,
        compute_time_ms: null,
        heavy_scorer_batches: 0,
      },
    };
    const pendingVisual = makeUnit("visual-xai-1", {
      source_type: "visual_observation",
      text: VISUAL_XAI_QA_OBSERVATION,
      frame_id: "F001",
      frame_ids: ["F001"],
      evidence_refs: ["F001"],
      producer: "Qwen/Qwen2.5-VL-7B-Instruct",
      eligible_for_frozen_g1: false,
      selection_score: null,
      logits: null,
      observation_type: "scene",
      evidence_frames: [pendingFrame],
    });
    const readyVisual = { ...pendingVisual, evidence_frames: [VISUAL_XAI_QA_FRAME] };
    const getResult = vi
      .spyOn(apiClient, "getJobResult")
      .mockResolvedValueOnce(makeResultResponse({ supplemental: [pendingVisual] }))
      .mockResolvedValueOnce(makeResultResponse({ supplemental: [readyVisual] }));
    const requestXAI = vi.spyOn(apiClient, "requestVisualXAI").mockResolvedValueOnce({
      api_version: "v1",
      job_id: JOB_ID,
      unit_id: "visual-xai-1",
      visual_xai: {
        status: "ready",
        profile: "public",
        grid_rows: 6,
        grid_columns: 6,
        attribution_batch_size: 2,
        configuration_fingerprint: "e".repeat(64),
        source_frame_count: 1,
        cache_hit: true,
        queue_wait_ms: 0,
        compute_time_ms: 0,
        heavy_scorer_batches: 0,
        unavailable_reason: null,
      },
      poll_after_ms: null,
    });

    renderResult();

    expect(await screen.findByText("FAKE")).toBeVisible();
    expect(screen.getByText("High-cost post-hoc attribution is available.")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Generate XAI" }));
    expect(await screen.findByText("Qwen occlusion attribution")).toBeVisible();
    await waitFor(() => expect(getResult).toHaveBeenCalledTimes(2));
    expect(requestXAI).toHaveBeenCalledWith(JOB_ID, "visual-xai-1");
    expect(screen.getByText("FAKE")).toBeVisible();
  });

  it("shows an explicit unavailable state for older visual results without XAI", async () => {
    const visual = makeUnit("visual-legacy", {
      source_type: "visual_observation",
      text: "A generic person stands near a stage.",
      frame_id: "F002",
      frame_ids: ["F002"],
      eligible_for_frozen_g1: false,
      selection_score: null,
      logits: null,
      evidence_frames: [
        {
          frame_id: "F002",
          frame_index: 30,
          timestamp: 1.2,
          original_image: null,
          annotated_image: null,
          bbox: null,
          regions: [],
          explanation: "Frame metadata only.",
        },
      ],
    });
    vi.spyOn(apiClient, "getJobResult").mockResolvedValueOnce(
      makeResultResponse({ supplemental: [visual] }),
    );

    renderResult();

    expect(await screen.findByText("Visual XAI unavailable")).toBeVisible();
    expect(screen.getByText("This older result does not contain an XAI artifact.")).toBeVisible();
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
