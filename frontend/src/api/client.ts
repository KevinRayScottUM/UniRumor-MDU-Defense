import type {
  HealthResponse,
  JobResultResponse,
  JobStatusResponse,
  JobSubmissionResponse,
  PublicErrorEnvelope,
  ReadinessResponse,
  SubmitJobInput,
} from "../types";

const API_PREFIX = "/api/v1";

export const API_ENDPOINTS = Object.freeze({
  health: `${API_PREFIX}/health`,
  readiness: `${API_PREFIX}/readiness`,
  jobs: `${API_PREFIX}/jobs`,
});

export interface ApiClientOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
}

interface ResponseContract<T> {
  acceptedStatuses?: readonly number[];
  isValid?: (body: unknown, status: number) => body is T;
}

export class ApiClientError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;

  constructor(
    status: number,
    code: string,
    message: string,
    requestId: string | null,
  ) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

function normalizeBaseUrl(value: string | undefined): string {
  const normalized = value?.trim() ?? "";
  return normalized.replace(/\/+$/, "");
}

function isPublicErrorEnvelope(value: unknown): value is PublicErrorEnvelope {
  if (typeof value !== "object" || value === null || !("error" in value)) {
    return false;
  }
  const error = value.error;
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    typeof error.code === "string" &&
    "message" in error &&
    typeof error.message === "string" &&
    "request_id" in error &&
    typeof error.request_id === "string"
  );
}

function isReadinessResponse(
  value: unknown,
  status: number,
): value is ReadinessResponse {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  if (
    !("api_version" in value) ||
    value.api_version !== "v1" ||
    !("status" in value) ||
    !("accepting_jobs" in value) ||
    typeof value.accepting_jobs !== "boolean" ||
    !("capacity_state" in value)
  ) {
    return false;
  }
  if (status === 200) {
    return (
      value.status === "ready" &&
      value.accepting_jobs &&
      value.capacity_state === "available"
    );
  }
  return (
    status === 503 &&
    value.status === "not_ready" &&
    !value.accepting_jobs &&
    (value.capacity_state === "full" || value.capacity_state === "unavailable")
  );
}

function jobPath(jobId: string): string {
  return `${API_ENDPOINTS.jobs}/${encodeURIComponent(jobId)}`;
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor({ baseUrl, fetch: fetchImpl = globalThis.fetch }: ApiClientOptions = {}) {
    if (typeof fetchImpl !== "function") {
      throw new TypeError("A Fetch API implementation is required.");
    }
    this.baseUrl = normalizeBaseUrl(baseUrl);
    this.fetchImpl = fetchImpl;
  }

  private async request<T>(
    path: string,
    init?: RequestInit,
    contract?: ResponseContract<T>,
  ): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...init?.headers,
      },
    });

    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      // A malformed or empty response is handled as a public-safe client error.
    }

    if (!response.ok && isPublicErrorEnvelope(body)) {
      throw new ApiClientError(
        response.status,
        body.error.code,
        body.error.message,
        body.error.request_id,
      );
    }
    const accepted =
      response.ok || contract?.acceptedStatuses?.includes(response.status) === true;
    if (!accepted || (contract?.isValid && !contract.isValid(body, response.status))) {
      throw new ApiClientError(
        response.status,
        "invalid_response",
        "The service returned an invalid response.",
        response.headers.get("X-Request-ID"),
      );
    }

    return body as T;
  }

  getHealth(): Promise<HealthResponse> {
    return this.request(API_ENDPOINTS.health);
  }

  getReadiness(): Promise<ReadinessResponse> {
    return this.request(API_ENDPOINTS.readiness, undefined, {
      acceptedStatuses: [503],
      isValid: isReadinessResponse,
    });
  }

  submitJob({ claim, video }: SubmitJobInput): Promise<JobSubmissionResponse> {
    const body = new FormData();
    body.append("claim", claim);
    body.append("video", video);
    return this.request(API_ENDPOINTS.jobs, { method: "POST", body });
  }

  getJob(jobId: string): Promise<JobStatusResponse> {
    return this.request(jobPath(jobId));
  }

  getJobResult(jobId: string): Promise<JobResultResponse> {
    return this.request(`${jobPath(jobId)}/result`);
  }
}

export function createApiClient(options?: ApiClientOptions): ApiClient {
  return new ApiClient(options);
}
