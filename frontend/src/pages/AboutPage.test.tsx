import { cleanup, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";

import { AppRoutes } from "../app/App";

function renderAboutPage() {
  return render(
    <MemoryRouter initialEntries={["/about"]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

afterEach(cleanup);

describe("About MDU research showcase", () => {
  it("explains the supported project concepts without result-like content", () => {
    renderAboutPage();

    for (const heading of [
      "Claim-centered verification",
      "Candidate units",
      "Evidence selection",
      "Explainable verification",
    ]) {
      expect(screen.getByRole("heading", { name: heading })).toBeVisible();
    }

    expect(screen.getByText("No result data")).toBeVisible();
    expect(
      screen.getByText(
        "Selection identifies explanation units; it does not replace the full eligible prediction pool.",
      ),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", {
        name: "The website presents; it does not predict.",
      }),
    ).toBeVisible();
  });

  it("renders the conceptual evidence hierarchy and complete workflow", () => {
    renderAboutPage();

    const concept = screen.getByLabelText(
      "Conceptual flow from a focal claim through candidate units to selected explanation units",
    );
    expect(within(concept).getByText("Text unit")).toBeVisible();
    expect(within(concept).getByText("Transcript unit")).toBeVisible();
    expect(within(concept).getByText("OCR unit")).toBeVisible();
    expect(within(concept).getAllByText("Selected for explanation")).toHaveLength(2);

    const workflow = screen.getByRole("list", {
      name: "UniRumor-MDU verification workflow",
    });
    expect(within(workflow).getAllByRole("listitem")).toHaveLength(4);
    expect(screen.getByRole("link", { name: /Start a verification/ })).toHaveAttribute(
      "href",
      "/#verify",
    );
  });
});
