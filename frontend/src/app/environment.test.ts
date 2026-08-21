import { describe, expect, it } from "vitest";

import { resolveApiBaseUrl } from "./environment";

describe("production environment configuration", () => {
  it("uses same-origin requests when the API base URL is unset", () => {
    expect(resolveApiBaseUrl(undefined)).toBe("");
    expect(resolveApiBaseUrl("   ")).toBe("");
  });

  it("normalizes an injected HTTP(S) API origin or gateway prefix", () => {
    expect(resolveApiBaseUrl(" https://api.example.test/ ")).toBe(
      "https://api.example.test",
    );
    expect(resolveApiBaseUrl("https://api.example.test/gateway/")).toBe(
      "https://api.example.test/gateway",
    );
  });

  it.each([
    "api.example.test",
    "/api",
    "file:///tmp/service",
    "https://user:secret@api.example.test",
    "https://api.example.test?internal=1",
    "https://api.example.test#private",
  ])("rejects unsafe or ambiguous configuration without echoing it: %s", (value) => {
    expect(() => resolveApiBaseUrl(value)).toThrowError(
      "VITE_API_BASE_URL must be an absolute HTTP(S) URL without credentials, a query, or a fragment.",
    );
  });
});
