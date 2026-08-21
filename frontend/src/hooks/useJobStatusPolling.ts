import { useEffect, useState } from "react";

import { ApiClientError } from "../api";
import { apiClient } from "../app/api";
import type { JobStatus } from "../types";

const FALLBACK_POLL_INTERVAL_MS = 3000;

export type UnavailableJobState = "expired" | "unknown";

export interface PublicPollingError {
  message: string;
  requestId?: string;
}

export interface UnavailableJob {
  state: UnavailableJobState;
  message: string;
  requestId?: string;
}

export interface JobStatusPollingResult {
  job?: JobStatus;
  loading: boolean;
  pollingError?: PublicPollingError;
  unavailable?: UnavailableJob;
  retry: () => void;
}

function isTerminal(state: JobStatus["state"]): boolean {
  return state === "completed" || state === "failed" || state === "expired";
}

function nextPollDelay(job: JobStatus): number {
  return typeof job.poll_after_ms === "number" && job.poll_after_ms > 0
    ? job.poll_after_ms
    : FALLBACK_POLL_INTERVAL_MS;
}

function unavailableFromError(error: unknown): UnavailableJob | undefined {
  if (!(error instanceof ApiClientError)) {
    return undefined;
  }
  if (error.code === "job_expired") {
    return {
      state: "expired",
      message: error.message,
      requestId: error.requestId ?? undefined,
    };
  }
  if (error.code === "job_not_found") {
    return {
      state: "unknown",
      message: error.message,
      requestId: error.requestId ?? undefined,
    };
  }
  return undefined;
}

function pollingErrorFrom(error: unknown): PublicPollingError {
  if (error instanceof ApiClientError) {
    return {
      message: error.message,
      requestId: error.requestId ?? undefined,
    };
  }
  return {
    message: "The current job status could not be retrieved. Automatic updates will retry.",
  };
}

export function useJobStatusPolling(
  jobId: string | undefined,
): JobStatusPollingResult {
  const [job, setJob] = useState<JobStatus>();
  const [loading, setLoading] = useState(true);
  const [pollingError, setPollingError] = useState<PublicPollingError>();
  const [unavailable, setUnavailable] = useState<UnavailableJob>();
  const [refreshVersion, setRefreshVersion] = useState(0);

  useEffect(() => {
    let disposed = false;
    let timer: number | undefined;

    setJob(undefined);
    setPollingError(undefined);
    setUnavailable(undefined);
    setLoading(true);

    if (!jobId) {
      setLoading(false);
      setUnavailable({
        state: "unknown",
        message: "Job not found.",
      });
      return () => {
        disposed = true;
      };
    }
    const requestedJobId = jobId;

    async function poll() {
      try {
        const response = await apiClient.getJob(requestedJobId);
        if (disposed) return;

        setJob(response.job);
        setPollingError(undefined);
        setUnavailable(undefined);
        setLoading(false);

        if (!isTerminal(response.job.state)) {
          timer = window.setTimeout(poll, nextPollDelay(response.job));
        }
      } catch (error) {
        if (disposed) return;

        const unavailableJob = unavailableFromError(error);
        setLoading(false);
        if (unavailableJob) {
          setJob(undefined);
          setPollingError(undefined);
          setUnavailable(unavailableJob);
          return;
        }

        setPollingError(pollingErrorFrom(error));
        timer = window.setTimeout(poll, FALLBACK_POLL_INTERVAL_MS);
      }
    }

    void poll();

    return () => {
      disposed = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [jobId, refreshVersion]);

  return {
    job,
    loading,
    pollingError,
    unavailable,
    retry: () => setRefreshVersion((version) => version + 1),
  };
}
