import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";

import { AppRoutes } from "./App";

function renderRoute(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

afterEach(cleanup);

describe("application shell", () => {
  it("renders the home page inside the application layout", () => {
    renderRoute("/");

    expect(
      screen.getByRole("link", { name: "UniRumor MDU Defense home" }),
    ).toBeVisible();
    expect(
      screen.getByRole("navigation", { name: "Primary navigation" }),
    ).toBeVisible();
    expect(
      screen.getAllByRole("link", {
        name: /^(Home|Verify|Sessions|About MDU)$/,
      }),
    ).toHaveLength(8);
    expect(
      screen.getByRole("heading", {
        name: "Explainable Multimodal Misinformation Verification",
      }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Start Verification" })).toBeDisabled();
    expect(
      screen.getByRole("heading", { name: "Move from a label to an evidence trail" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "A unit-level view of verification" }),
    ).toBeVisible();
  });

  it("renders the About MDU research route", () => {
    renderRoute("/about");

    expect(
      screen.getByRole("heading", { name: "What is a Minimal Deceptive Unit?" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Evidence that remains traceable" }),
    ).toBeVisible();
    expect(screen.getAllByRole("link", { name: "About MDU" })).toHaveLength(2);
    expect(
      screen.getAllByRole("link", { name: "About MDU" })[0],
    ).toHaveAttribute("aria-current", "page");
    expect(document.title).toBe(
      "About MDU | UniRumor-MDU: Explainable Multimodal Misinformation Verification",
    );
  });

  it("renders the illustrative demo route with an explicit disclosure", () => {
    renderRoute("/demo");

    expect(
      screen.getByRole("heading", {
        name: "See how an explainable result is organized",
      }),
    ).toBeVisible();
    expect(
      screen.getByText("Illustrative demo only. Not a live model result."),
    ).toBeVisible();
    expect(document.title).toBe(
      "Illustrative Demo | UniRumor-MDU: Explainable Multimodal Misinformation Verification",
    );
  });

  it("renders job monitoring and result routes", () => {
    const status = renderRoute("/jobs/job_123");
    expect(
      screen.getByRole("heading", { name: "Verification Session" }),
    ).toBeVisible();
    expect(screen.getByText("job_123")).toBeVisible();
    status.unmount();

    renderRoute("/jobs/job_123/result");
    expect(screen.getByRole("heading", { name: "Verification Result" })).toBeVisible();
    expect(screen.getByText("job_123")).toBeVisible();
  });
});
