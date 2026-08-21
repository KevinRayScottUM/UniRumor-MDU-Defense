import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClientError } from "../api";
import { AppRoutes } from "../app/App";
import { apiClient } from "../app/api";
import type { JobState, JobStatus, JobStatusResponse } from "../types";

const JOB_ID = "job_0123456789abcdef0123456789abcdef";

function makeJob(
  state: JobState,
  overrides: Partial<JobStatus> = {},
): JobStatusResponse {
  const terminal = state === "completed" || state === "failed";
  return {
    api_version: "v1",
    job: {
      job_id: JOB_ID,
      state,
      queue_position: state === "queued" ? 2 : null,
      created_at: "2026-08-21T01:00:00Z",
      started_at:
        state === "running" || terminal ? "2026-08-21T01:00:03Z" : null,
      finished_at: terminal ? "2026-08-21T01:00:08Z" : null,
      expires_at: terminal ? "2026-08-21T01:15:08Z" : null,
      queue_elapsed_ms: state === "accepted" ? 0 : 3000,
      execution_elapsed_ms: terminal ? 5000 : state === "running" ? 1200 : 0,
      failure: null,
      links: {
        self: `/api/v1/jobs/${JOB_ID}`,
        result: `/api/v1/jobs/${JOB_ID}/result`,
      },
      poll_after_ms: terminal ? null : 25,
      ...overrides,
    },
  };
}

function renderJobStatus() {
  return render(
    <MemoryRouter initialEntries={[`/jobs/${JOB_ID}`]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("job monitoring experience", () => {
  it("polls authoritative states and stops after completion", async () => {
    const getJob = vi
      .spyOn(apiClient, "getJob")
      .mockResolvedValueOnce(makeJob("queued", { poll_after_ms: 100 }))
      .mockResolvedValueOnce(makeJob("running"))
      .mockResolvedValueOnce(makeJob("completed"));
    renderJobStatus();

    expect(await screen.findByText("Queued")).toBeVisible();
    expect(screen.getByText("Queue position")).toBeVisible();
    await waitFor(() => expect(getJob).toHaveBeenCalledTimes(3));
    expect(await screen.findByText("Completed")).toBeVisible();
    expect(
      screen.getByRole("link", { name: "View result" }),
    ).toHaveAttribute("href", `/jobs/${JOB_ID}/result`);
    expect(screen.queryByText("Automatic updates active")).not.toBeInTheDocument();

    await new Promise((resolve) => window.setTimeout(resolve, 70));
    expect(getJob).toHaveBeenCalledTimes(3);
  });

  it("cleans up the next polling timer when the page unmounts", async () => {
    const getJob = vi
      .spyOn(apiClient, "getJob")
      .mockResolvedValue(makeJob("queued", { poll_after_ms: 30 }));
    const view = renderJobStatus();

    expect(await screen.findByText("Queued")).toBeVisible();
    expect(getJob).toHaveBeenCalledTimes(1);
    view.unmount();
    await new Promise((resolve) => window.setTimeout(resolve, 70));

    expect(getJob).toHaveBeenCalledTimes(1);
  });

  it("maps public expired and unknown errors without polling again", async () => {
    const getJob = vi.spyOn(apiClient, "getJob").mockRejectedValueOnce(
      new ApiClientError(
        410,
        "job_expired",
        "Job has expired.",
        "req_0123456789abcdef0123456789abcdef",
      ),
    );
    const expired = renderJobStatus();

    expect(
      await screen.findByRole("heading", { name: "This session has expired" }),
    ).toBeVisible();
    expect(screen.getByText("Expired")).toBeVisible();
    expect(screen.getByText(/Request ID: req_0123/)).toBeVisible();
    expired.unmount();

    getJob.mockRejectedValueOnce(
      new ApiClientError(
        404,
        "job_not_found",
        "Job not found.",
        "req_abcdef0123456789abcdef0123456789",
      ),
    );
    renderJobStatus();

    expect(
      await screen.findByRole("heading", { name: "Job not found" }),
    ).toBeVisible();
    expect(screen.getByText("Unknown")).toBeVisible();
    expect(getJob).toHaveBeenCalledTimes(2);
  });

  it("recovers from a network failure through the explicit retry action", async () => {
    const getJob = vi
      .spyOn(apiClient, "getJob")
      .mockRejectedValueOnce(new TypeError("network unavailable"))
      .mockResolvedValueOnce(makeJob("completed"));
    renderJobStatus();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The current job status could not be retrieved.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry status" }));

    expect(await screen.findByText("Completed")).toBeVisible();
    expect(getJob).toHaveBeenCalledTimes(2);
  });

  it("renders only the backend failure message and incident identifier", async () => {
    vi.spyOn(apiClient, "getJob").mockResolvedValueOnce(
      makeJob("failed", {
        failure: {
          code: "runtime_execution_failed",
          message: "Verification execution failed.",
          incident_id: "incident_0123456789abcdef0123456789abcdef",
        },
      }),
    );
    renderJobStatus();

    expect((await screen.findAllByText("Failed")).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("Verification execution failed.")).toBeVisible();
    expect(
      screen.getByText(/incident_0123456789abcdef0123456789abcdef/),
    ).toBeVisible();
    expect(screen.getByText("Unavailable")).toBeVisible();
  });
});
