import {
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
} from "react";
import { useNavigate } from "react-router-dom";

import { ApiClientError } from "../api";
import { apiClient } from "../app/api";
import { Badge, Button, Card, ErrorMessage, TextArea } from "./ui";

const MAX_CLAIM_LENGTH = 2000;
const ACCEPTED_VIDEO_TYPES = Object.freeze({
  ".mp4": ["video/mp4"],
  ".m4v": ["video/x-m4v", "video/mp4"],
  ".mov": ["video/quicktime"],
  ".webm": ["video/webm"],
} as const);

interface SubmissionError {
  message: string;
  requestId?: string;
}

function claimValidationMessage(claim: string): string | undefined {
  if (!claim.trim()) {
    return "Enter the exact claim you want to verify.";
  }
  if (claim.length > MAX_CLAIM_LENGTH) {
    return `Keep the claim within ${MAX_CLAIM_LENGTH.toLocaleString()} characters.`;
  }
  return undefined;
}

function videoValidationMessage(file: File): string | undefined {
  const extensionIndex = file.name.lastIndexOf(".");
  const extension = extensionIndex >= 0 ? file.name.slice(extensionIndex).toLowerCase() : "";
  const acceptedMimes = ACCEPTED_VIDEO_TYPES[
    extension as keyof typeof ACCEPTED_VIDEO_TYPES
  ];

  if (
    !acceptedMimes ||
    !(acceptedMimes as readonly string[]).includes(file.type)
  ) {
    return "Choose an MP4, M4V, MOV, or WebM video with a matching file type.";
  }
  if (file.size === 0) {
    return "The selected video is empty.";
  }
  return undefined;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function toSubmissionError(error: unknown): SubmissionError {
  if (error instanceof ApiClientError) {
    return {
      message: error.message,
      requestId: error.requestId ?? undefined,
    };
  }
  return {
    message: "The verification service could not be reached. Please try again.",
  };
}

export function VerificationSubmissionCard() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [claim, setClaim] = useState("");
  const [claimTouched, setClaimTouched] = useState(false);
  const [video, setVideo] = useState<File | null>(null);
  const [videoError, setVideoError] = useState<string>();
  const [dragActive, setDragActive] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submissionError, setSubmissionError] = useState<SubmissionError>();

  const claimError = claimValidationMessage(claim);
  const formReady = !claimError && video !== null && !videoError && !submitting;

  function selectVideo(file: File | undefined) {
    setSubmissionError(undefined);
    if (!file) {
      setVideo(null);
      setVideoError("Choose one video to verify.");
      return;
    }

    const nextError = videoValidationMessage(file);
    if (nextError) {
      setVideo(null);
      setVideoError(nextError);
      if (inputRef.current) {
        inputRef.current.value = "";
      }
      return;
    }

    setVideo(file);
    setVideoError(undefined);
  }

  function handleVideoChange(event: ChangeEvent<HTMLInputElement>) {
    selectVideo(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    if (submitting) {
      return;
    }
    if (event.dataTransfer.files.length !== 1) {
      setVideo(null);
      setVideoError("Choose exactly one video file.");
      return;
    }
    selectVideo(event.dataTransfer.files[0]);
  }

  function removeVideo() {
    setVideo(null);
    setVideoError(undefined);
    setSubmissionError(undefined);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setClaimTouched(true);
    setSubmissionError(undefined);

    if (claimError || !video || videoError || submitting) {
      if (!video && !videoError) {
        setVideoError("Choose one video to verify.");
      }
      return;
    }

    setSubmitting(true);
    try {
      const submission = await apiClient.submitJob({ claim, video });
      navigate(`/jobs/${encodeURIComponent(submission.job_id)}`);
    } catch (error) {
      setSubmissionError(toSubmissionError(error));
      setSubmitting(false);
    }
  }

  return (
    <Card className="verification-card" variant="glass">
      <div className="verification-card__topline">
        <div>
          <p className="verification-card__step">Verification request</p>
          <h2>Provide one focal claim and one source video</h2>
        </div>
        <Badge tone="info" withDot>
          Server validation
        </Badge>
      </div>

      <form className="verification-form" noValidate onSubmit={handleSubmit}>
        <div className="verification-field">
          <TextArea
            aria-describedby="claim-character-count"
            disabled={submitting}
            error={claimTouched ? claimError : undefined}
            hint="Use the exact wording of the claim. The text is preserved for the verification request."
            label="Claim to verify"
            maxLength={MAX_CLAIM_LENGTH + 1}
            name="claim"
            onBlur={() => setClaimTouched(true)}
            onChange={(event) => {
              setClaim(event.target.value);
              setSubmissionError(undefined);
            }}
            placeholder="Example: The video shows a current event taking place in the stated location."
            rows={5}
            required
            value={claim}
          />
          <p
            aria-live="polite"
            className={
              claimError && claimTouched
                ? "character-count character-count--error"
                : "character-count"
            }
            id="claim-character-count"
          >
            {claim.length.toLocaleString()} / {MAX_CLAIM_LENGTH.toLocaleString()}
          </p>
        </div>

        <fieldset className="video-fieldset" disabled={submitting}>
          <legend>Source video</legend>
          <p className="video-fieldset__hint" id="video-upload-hint">
            MP4, M4V, MOV, or WebM. Upload limits and file contents are validated by the server.
          </p>
          <div
            className={[
              "video-dropzone",
              dragActive ? "video-dropzone--active" : "",
              videoError ? "video-dropzone--error" : "",
              video ? "video-dropzone--selected" : "",
              submitting ? "video-dropzone--uploading" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            data-testid="video-dropzone"
            onDragEnter={(event) => {
              event.preventDefault();
              if (!submitting) setDragActive(true);
            }}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                setDragActive(false);
              }
            }}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
            }}
            onDrop={handleDrop}
          >
            <input
              ref={inputRef}
              accept=".mp4,.m4v,.mov,.webm,video/mp4,video/x-m4v,video/quicktime,video/webm"
              aria-describedby={
                videoError
                  ? "video-upload-hint video-upload-error"
                  : "video-upload-hint"
              }
              aria-invalid={Boolean(videoError) || undefined}
              aria-label="Video file"
              className="video-dropzone__input"
              id="verification-video"
              name="video"
              onChange={handleVideoChange}
              required
              type="file"
            />
            <label className="video-dropzone__label" htmlFor="verification-video">
              <span aria-hidden="true" className="video-dropzone__icon">
                {submitting ? <span className="ui-spinner" /> : "↑"}
              </span>
              <span className="video-dropzone__copy">
                <strong>
                  {submitting
                    ? "Uploading securely"
                    : video
                      ? video.name
                      : "Drop a video here or browse"}
                </strong>
                <span>
                  {video
                    ? `${formatFileSize(video.size)} · ready for server validation`
                    : "MP4, M4V, MOV, or WebM · upload limits are enforced by the server"}
                </span>
              </span>
            </label>
            {video && !submitting ? (
              <Button onClick={removeVideo} size="small" variant="ghost">
                Remove
              </Button>
            ) : null}
          </div>
          {videoError ? (
            <p className="video-fieldset__error" id="video-upload-error" role="alert">
              {videoError}
            </p>
          ) : null}
        </fieldset>

        {submissionError ? (
          <ErrorMessage
            message={submissionError.message}
            requestId={submissionError.requestId}
            title="Verification could not start"
          />
        ) : null}

        <div className="verification-form__footer">
          <p>
            Submission creates a server-managed job. No prediction is produced
            in the browser.
          </p>
          <Button
            className="verification-form__submit"
            disabled={!formReady}
            isLoading={submitting}
            size="large"
            type="submit"
          >
            {submitting ? "Submitting verification" : "Start Verification"}
          </Button>
        </div>
      </form>
    </Card>
  );
}
