import { useEffect, useState } from "react";

import { ApiClientError } from "../api";
import { apiClient } from "../app/api";
import type { JobResultResponse } from "../types";

export type UnavailableResultState =
  | "not_completed"
  | "failed"
  | "expired"
  | "unknown";

export interface PublicResultError {
  message: string;
  requestId?: string;
}

export interface UnavailableResult extends PublicResultError {
  state: UnavailableResultState;
}

export interface JobResultLoadingState {
  response?: JobResultResponse;
  loading: boolean;
  error?: PublicResultError;
  unavailable?: UnavailableResult;
  retry: () => void;
}

function unavailableFromError(error: unknown): UnavailableResult | undefined {
  if (!(error instanceof ApiClientError)) return undefined;

  const shared = {
    message: error.message,
    requestId: error.requestId ?? undefined,
  };
  if (error.code === "job_not_completed") {
    return { state: "not_completed", ...shared };
  }
  if (error.code === "job_failed") {
    return { state: "failed", ...shared };
  }
  if (error.code === "job_expired") {
    return { state: "expired", ...shared };
  }
  if (error.code === "job_not_found") {
    return { state: "unknown", ...shared };
  }
  return undefined;
}

function publicErrorFrom(error: unknown): PublicResultError {
  if (error instanceof ApiClientError) {
    return {
      message: error.message,
      requestId: error.requestId ?? undefined,
    };
  }
  return {
    message: "The verification result could not be retrieved. Please try again.",
  };
}

export function useJobResult(jobId: string | undefined): JobResultLoadingState {
  const [response, setResponse] = useState<JobResultResponse>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<PublicResultError>();
  const [unavailable, setUnavailable] = useState<UnavailableResult>();
  const [refreshVersion, setRefreshVersion] = useState(0);

  useEffect(() => {
    let disposed = false;

    setResponse(undefined);
    setError(undefined);
    setUnavailable(undefined);
    setLoading(true);

    if (!jobId) {
      setLoading(false);
      setUnavailable({ state: "unknown", message: "Job not found." });
      return () => {
        disposed = true;
      };
    }
    const requestedJobId = jobId;

    async function loadResult() {
      try {
        const nextResponse = await apiClient.getJobResult(requestedJobId);
        if (disposed) return;

        setResponse(nextResponse);
        setLoading(false);
      } catch (caught) {
        if (disposed) return;

        const unavailableResult = unavailableFromError(caught);
        setLoading(false);
        if (unavailableResult) {
          setUnavailable(unavailableResult);
          return;
        }
        setError(publicErrorFrom(caught));
      }
    }

    void loadResult();

    return () => {
      disposed = true;
    };
  }, [jobId, refreshVersion]);

  return {
    response,
    loading,
    error,
    unavailable,
    retry: () => setRefreshVersion((version) => version + 1),
  };
}
