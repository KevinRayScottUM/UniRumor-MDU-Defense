import { describe, expect, it, vi } from "vitest";

import { API_ENDPOINTS, ApiClientError, createApiClient } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("ApiClient", () => {
  it("constructs with a normalized public API base URL", () => {
    const client = createApiClient({
      baseUrl: " https://api.example.test/ ",
      fetch: vi.fn(),
    });

    expect(client.baseUrl).toBe("https://api.example.test");
    expect(API_ENDPOINTS).toEqual({
      health: "/api/v1/health",
      readiness: "/api/v1/readiness",
      jobs: "/api/v1/jobs",
    });
  });

  it("invokes the injected Fetch implementation without rebinding it", async () => {
    let observedThis: unknown = "not-called";
    const fetchMock = function (this: unknown) {
      observedThis = this;
      return Promise.resolve(jsonResponse({ api_version: "v1", status: "ok" }));
    } as typeof fetch;
    const client = createApiClient({ fetch: fetchMock });

    await client.getHealth();

    expect(observedThis).toBeUndefined();
  });

  it("uses the authoritative health, readiness, status, and result routes", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ api_version: "v1", status: "ok" }))
      .mockResolvedValueOnce(
        jsonResponse({
          api_version: "v1",
          status: "ready",
          accepting_jobs: true,
          capacity_state: "available",
        }),
      )
      .mockResolvedValue(jsonResponse({ api_version: "v1" }));
    const client = createApiClient({ fetch: fetchMock });

    await client.getHealth();
    await client.getReadiness();
    await client.getJob("job_test/value");
    await client.getJobResult("job_test/value");
    await client.requestVisualXAI("job_test/value", "visual/unit");
    await client.getVisualXAI("job_test/value", "visual/unit");

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/v1/health",
      "/api/v1/readiness",
      "/api/v1/jobs/job_test%2Fvalue",
      "/api/v1/jobs/job_test%2Fvalue/result",
      "/api/v1/jobs/job_test%2Fvalue/visual-xai/visual%2Funit",
      "/api/v1/jobs/job_test%2Fvalue/visual-xai/visual%2Funit",
    ]);
    expect(fetchMock.mock.calls[4][1]?.method).toBe("POST");
  });

  it("returns the valid readiness payload for HTTP 200", async () => {
    const readiness = {
      api_version: "v1",
      status: "ready",
      accepting_jobs: true,
      capacity_state: "available",
    } as const;
    const client = createApiClient({
      fetch: vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(readiness, 200)),
    });

    await expect(client.getReadiness()).resolves.toEqual(readiness);
  });

  it("returns the valid not-ready/full payload for HTTP 503", async () => {
    const readiness = {
      api_version: "v1",
      status: "not_ready",
      accepting_jobs: false,
      capacity_state: "full",
    } as const;
    const client = createApiClient({
      fetch: vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(readiness, 503)),
    });

    await expect(client.getReadiness()).resolves.toEqual(readiness);
  });

  it("rejects malformed readiness payloads at accepted status codes", async () => {
    const client = createApiClient({
      fetch: vi.fn<typeof fetch>().mockResolvedValue(
        jsonResponse(
          {
            api_version: "v1",
            status: "ready",
            accepting_jobs: false,
            capacity_state: "full",
          },
          503,
        ),
      ),
    });

    await expect(client.getReadiness()).rejects.toMatchObject({
      name: "ApiClientError",
      status: 503,
      code: "invalid_response",
    } satisfies Partial<ApiClientError>);
  });

  it("submits exactly the claim and video through backend multipart validation", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse(
        {
          api_version: "v1",
          job_id: "job_0123456789abcdef0123456789abcdef",
          state: "queued",
          request_id: "req_0123456789abcdef0123456789abcdef",
        },
        202,
      ),
    );
    const client = createApiClient({ fetch: fetchMock });
    const video = new File(["video"], "sample.mp4", { type: "video/mp4" });

    await client.submitJob({ claim: "Exact focal claim", video });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/jobs");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeInstanceOf(FormData);
    const body = init?.body as FormData;
    expect([...body.keys()]).toEqual(["claim", "video"]);
    expect(body.get("claim")).toBe("Exact focal claim");
    expect(body.get("video")).toBe(video);
    expect(new Headers(init?.headers).has("Content-Type")).toBe(false);
  });

  it("preserves only the public backend error contract", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse(
        {
          api_version: "v1",
          error: {
            code: "queue_full",
            message: "Job queue is full.",
            request_id: "req_0123456789abcdef0123456789abcdef",
          },
        },
        429,
      ),
    );
    const client = createApiClient({ fetch: fetchMock });

    await expect(client.getReadiness()).rejects.toMatchObject({
      name: "ApiClientError",
      status: 429,
      code: "queue_full",
      message: "Job queue is full.",
      requestId: "req_0123456789abcdef0123456789abcdef",
    } satisfies Partial<ApiClientError>);
  });
});
