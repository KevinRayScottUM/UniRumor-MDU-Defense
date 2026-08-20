import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { AppRoutes } from "./App";

function renderRoute(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

describe("application shell", () => {
  it("renders the home page inside the application layout", () => {
    renderRoute("/");

    expect(screen.getByRole("link", { name: "UniRumor MDU Defense" })).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Check a focal claim against a video" }),
    ).toBeVisible();
  });

  it("renders job status and result placeholders from routes", () => {
    const status = renderRoute("/jobs/job_123");
    expect(screen.getByRole("heading", { name: "Verification job" })).toBeVisible();
    expect(screen.getByText("Job identifier: job_123")).toBeVisible();
    status.unmount();

    renderRoute("/jobs/job_123/result");
    expect(screen.getByRole("heading", { name: "Verification result" })).toBeVisible();
    expect(screen.getByText("Job identifier: job_123")).toBeVisible();
  });
});

