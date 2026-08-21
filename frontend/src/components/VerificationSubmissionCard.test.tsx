import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClientError } from "../api";
import { AppRoutes } from "../app/App";
import { apiClient } from "../app/api";

const FOCAL_CLAIM =
  "The video shows the stated event in the stated location.";

function renderHome() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

function completeForm() {
  fireEvent.change(screen.getByRole("textbox", { name: "Claim to verify" }), {
    target: { value: FOCAL_CLAIM },
  });
  const video = new File(["video-content"], "source.mp4", {
    type: "video/mp4",
  });
  fireEvent.change(screen.getByLabelText("Video file"), {
    target: { files: [video] },
  });
  return video;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("verification submission experience", () => {
  it("keeps submission disabled until the public inputs are valid", () => {
    renderHome();

    const submit = screen.getByRole("button", { name: "Start Verification" });
    expect(submit).toBeDisabled();
    expect(screen.getByText("0 / 2,000")).toBeVisible();

    const video = completeForm();

    expect(submit).toBeEnabled();
    expect(screen.getByText(video.name)).toBeVisible();
    expect(
      screen.getByText(`${FOCAL_CLAIM.length.toLocaleString()} / 2,000`),
    ).toBeVisible();
  });

  it("rejects an unsupported dropped file before submission", () => {
    renderHome();
    const unsupported = new File(["not-video"], "source.txt", {
      type: "text/plain",
    });

    fireEvent.drop(screen.getByTestId("video-dropzone"), {
      dataTransfer: { files: [unsupported] },
    });

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Choose an MP4, M4V, MOV, or WebM video",
    );
    expect(screen.getByRole("button", { name: "Start Verification" })).toBeDisabled();
  });

  it("submits through the existing API client and navigates with the returned job id", async () => {
    const submission = {
      api_version: "v1",
      job_id: "job_0123456789abcdef0123456789abcdef",
      state: "queued",
      request_id: "req_0123456789abcdef0123456789abcdef",
    } as const;
    const submitJob = vi.spyOn(apiClient, "submitJob").mockResolvedValue(submission);
    renderHome();
    const video = completeForm();

    fireEvent.click(screen.getByRole("button", { name: "Start Verification" }));

    await waitFor(() =>
      expect(submitJob).toHaveBeenCalledWith({
        claim: FOCAL_CLAIM,
        video,
      }),
    );
    expect(
      await screen.findByRole("heading", { name: "Verification Session" }),
    ).toBeVisible();
    expect(screen.getByText(submission.job_id)).toBeVisible();
  });

  it("locks the form and exposes upload and submission progress while pending", async () => {
    const submission = {
      api_version: "v1",
      job_id: "job_0123456789abcdef0123456789abcdef",
      state: "queued",
      request_id: "req_0123456789abcdef0123456789abcdef",
    } as const;
    let resolveSubmission: (value: typeof submission) => void = () => undefined;
    vi.spyOn(apiClient, "submitJob").mockReturnValue(
      new Promise((resolve) => {
        resolveSubmission = resolve;
      }),
    );
    renderHome();
    completeForm();

    fireEvent.click(screen.getByRole("button", { name: "Start Verification" }));

    expect(
      screen.getByRole("button", { name: "Submitting verification" }),
    ).toBeDisabled();
    expect(screen.getByText("Uploading securely")).toBeVisible();
    expect(screen.getByRole("textbox", { name: "Claim to verify" })).toBeDisabled();

    resolveSubmission(submission);
    expect(
      await screen.findByRole("heading", { name: "Verification Session" }),
    ).toBeVisible();
  });

  it("renders only the sanitized public API error and request id", async () => {
    vi.spyOn(apiClient, "submitJob").mockRejectedValue(
      new ApiClientError(
        429,
        "queue_full",
        "Job queue is full.",
        "req_0123456789abcdef0123456789abcdef",
      ),
    );
    renderHome();
    completeForm();

    fireEvent.click(screen.getByRole("button", { name: "Start Verification" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Job queue is full.");
    expect(alert).toHaveTextContent(
      "Request ID: req_0123456789abcdef0123456789abcdef",
    );
  });
});
