import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Badge, Button, ErrorMessage, Input, LoadingState, SectionHeader } from ".";

describe("design system primitives", () => {
  it("provides accessible button and status semantics", () => {
    render(
      <div>
        <Button isLoading>Submitting</Button>
        <Badge tone="success" withDot>
          Ready
        </Badge>
        <LoadingState detail="Waiting for the public API" label="Loading job" />
      </div>,
    );

    expect(screen.getByRole("button", { name: "Submitting" })).toBeDisabled();
    expect(screen.getByText("Ready")).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Loading job");
  });

  it("associates field errors with their native input", () => {
    render(
      <div>
        <Input
          aria-describedby="external-field-help"
          error="Enter a focal claim."
          label="Focal claim"
          name="claim"
        />
        <p id="external-field-help">External field guidance.</p>
      </div>,
    );

    const input = screen.getByRole("textbox", { name: "Focal claim" });
    expect(input).toBeInvalid();
    expect(input).toHaveAccessibleDescription(
      "External field guidance. Enter a focal claim.",
    );
  });

  it("exposes meaningful section and error structure", () => {
    render(
      <div>
        <SectionHeader
          description="Public-safe supporting copy."
          headingLevel={2}
          title="Evidence summary"
        />
        <ErrorMessage message="Verification could not be completed." />
      </div>,
    );

    expect(screen.getByRole("heading", { name: "Evidence summary" })).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Verification could not be completed.",
    );
  });
});
