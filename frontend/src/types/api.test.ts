import { describe, expect, expectTypeOf, it } from "vitest";

import type {
  HealthResponse,
  JobResultResponse,
  JobStatusResponse,
  PublicEvidenceFrame,
  PublicEvidenceUnit,
  PublicErrorEnvelope,
  ReadinessResponse,
} from "./api";

describe("public API types", () => {
  it("represent the backend health, readiness, status, result, and error envelopes", () => {
    const health = { api_version: "v1", status: "ok" } satisfies HealthResponse;
    const readiness = {
      api_version: "v1",
      status: "ready",
      accepting_jobs: true,
      capacity_state: "available",
    } satisfies ReadinessResponse;
    const error = {
      api_version: "v1",
      error: {
        code: "job_not_found",
        message: "Job not found.",
        request_id: "req_0123456789abcdef0123456789abcdef",
      },
    } satisfies PublicErrorEnvelope;

    expect(health.status).toBe("ok");
    expect(readiness.accepting_jobs).toBe(true);
    expect(error.error.code).toBe("job_not_found");
    expectTypeOf<JobStatusResponse["job"]["state"]>().toMatchTypeOf<
      "accepted" | "queued" | "running" | "completed" | "failed" | "expired"
    >();
    expectTypeOf<JobResultResponse["outcome"]["status"]>().toEqualTypeOf<"success">();
    expectTypeOf<NonNullable<PublicEvidenceUnit["evidence_frames"]>[number]>()
      .toEqualTypeOf<PublicEvidenceFrame>();
  });
});
