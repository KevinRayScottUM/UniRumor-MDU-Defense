import { cleanup, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppRoutes } from "../app/App";
import { apiClient } from "../app/api";

function renderDemoPage() {
  return render(
    <MemoryRouter initialEntries={["/demo"]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("illustrative research demo", () => {
  it("renders a clearly non-live claim, verdict, and MDU hierarchy", () => {
    renderDemoPage();

    expect(
      screen.getByText("Illustrative demo only. Not a live model result."),
    ).toBeVisible();
    expect(
      screen.getByText(
        "All claim and unit text below is interface copy. It is not extracted evidence, a prediction, or a scientific finding.",
      ),
    ).toBeVisible();
    expect(
      screen.getByLabelText(
        "Illustrative FAKE verdict presentation; not a live result",
      ),
    ).toHaveTextContent("FAKE");

    const candidates = screen
      .getByRole("heading", { name: "Candidate units" })
      .closest("section");
    const selected = screen
      .getByRole("heading", { name: "Selected units" })
      .closest("section");
    expect(candidates).not.toBeNull();
    expect(selected).not.toBeNull();
    expect(within(candidates as HTMLElement).getAllByRole("listitem")).toHaveLength(3);
    expect(within(selected as HTMLElement).getAllByRole("listitem")).toHaveLength(2);
  });

  it("does not contact any production API endpoint", () => {
    const health = vi.spyOn(apiClient, "getHealth");
    const readiness = vi.spyOn(apiClient, "getReadiness");
    const status = vi.spyOn(apiClient, "getJob");
    const result = vi.spyOn(apiClient, "getJobResult");
    const submit = vi.spyOn(apiClient, "submitJob");

    renderDemoPage();

    expect(health).not.toHaveBeenCalled();
    expect(readiness).not.toHaveBeenCalled();
    expect(status).not.toHaveBeenCalled();
    expect(result).not.toHaveBeenCalled();
    expect(submit).not.toHaveBeenCalled();
  });
});
