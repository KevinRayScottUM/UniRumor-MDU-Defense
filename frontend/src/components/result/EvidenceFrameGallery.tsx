import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type SyntheticEvent,
} from "react";
import { createPortal } from "react-dom";

import { apiClient } from "../../app/api";
import type {
  EvidenceSourceType,
  PublicEvidenceFrame,
  PublicEvidenceRegion,
  PublicVisualXAIMap,
  VisualXAIStatus,
} from "../../types";

export interface EvidenceFrameGalleryProps {
  frames: PublicEvidenceFrame[];
  jobId?: string;
  onVisualXAIReady?: () => void;
  sourceType: EvidenceSourceType;
  unitId?: string;
}

type ViewerXAIState = VisualXAIStatus | "legacy_missing";

interface ImageDimensions {
  width: number;
  height: number;
}

function frameName(frame: PublicEvidenceFrame, index: number): string {
  if (frame.frame_id) return frame.frame_id;
  if (frame.frame_index !== null) return `Frame ${String(frame.frame_index)}`;
  return `Referenced frame ${String(index + 1)}`;
}

function frameTime(frame: PublicEvidenceFrame): string | undefined {
  return frame.timestamp === null ? undefined : `${String(frame.timestamp)} s`;
}

function safeImageSource(value: string | null): string | undefined {
  if (!value) return undefined;
  if (/^data:image\/(?:jpeg|png|webp);base64,[a-z0-9+/=]+$/i.test(value)) {
    return value;
  }
  if (value.startsWith("/") || /^https:\/\//i.test(value)) return value;
  return undefined;
}

function xaiMethodLabel(method: string): string {
  if (method === "qwen_occlusion_logprob_v1") return "Qwen occlusion attribution";
  if (method === "siglip_semantic_grounding_v1") return "SigLIP semantic grounding";
  return "Post-hoc visual attribution";
}

function defaultAttributionMap(frame: PublicEvidenceFrame): PublicVisualXAIMap | undefined {
  const maps = frame.xai?.attribution_maps ?? [];
  return maps.find((item) => item.scope === "observation") ?? maps[0];
}

function regionRect(
  region: PublicEvidenceRegion,
  dimensions: ImageDimensions,
): { x: number; y: number; width: number; height: number } | undefined {
  if (region.bbox.length !== 4) return undefined;
  let [x1, y1, x2, y2] = region.bbox;
  if (![x1, y1, x2, y2].every(Number.isFinite)) return undefined;
  if (Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1) {
    x1 *= dimensions.width;
    x2 *= dimensions.width;
    y1 *= dimensions.height;
    y2 *= dimensions.height;
  }
  x1 = Math.max(0, Math.min(dimensions.width, x1));
  x2 = Math.max(0, Math.min(dimensions.width, x2));
  y1 = Math.max(0, Math.min(dimensions.height, y1));
  y2 = Math.max(0, Math.min(dimensions.height, y2));
  if (x2 <= x1 || y2 <= y1) return undefined;
  return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
}

function regionLabel(region: PublicEvidenceRegion, index: number): string {
  const label = region.text?.trim();
  return label ? label.slice(0, 42) : `OCR region ${String(index + 1)}`;
}

function AnnotatedFrame({
  annotated,
  frame,
  label,
  sourceOverride,
}: {
  annotated: boolean;
  frame: PublicEvidenceFrame;
  label: string;
  sourceOverride?: string;
}) {
  const [dimensions, setDimensions] = useState<ImageDimensions>();
  const original = safeImageSource(frame.original_image);
  const serverAnnotated = safeImageSource(frame.annotated_image);
  const override = safeImageSource(sourceOverride ?? null);
  const source = override ?? (annotated ? serverAnnotated ?? original : original);
  const shouldDrawRegions = annotated && !override && !serverAnnotated;
  const frameLevelLabel = frame.explanation.includes("OCR region unavailable")
    ? "OCR region unavailable"
    : "Frame-level evidence";

  function recordDimensions(event: SyntheticEvent<HTMLImageElement>) {
    const image = event.currentTarget;
    if (image.naturalWidth > 0 && image.naturalHeight > 0) {
      setDimensions({ width: image.naturalWidth, height: image.naturalHeight });
    }
  }

  if (!source) {
    return (
      <div className="evidence-frame__unavailable" role="img" aria-label={`${label} image unavailable`}>
        <span aria-hidden="true">▧</span>
        <strong>Frame image unavailable</strong>
        <small>Metadata remains available below.</small>
      </div>
    );
  }

  return (
    <div className="evidence-frame__image-wrap">
      <img
        alt={`${label}${override ? " XAI attribution" : annotated ? " annotated evidence" : " original"}`}
        onLoad={recordDimensions}
        src={source}
      />
      {shouldDrawRegions && dimensions && frame.regions.length > 0 ? (
        <svg
          aria-label={`${String(frame.regions.length)} grounded OCR region${frame.regions.length === 1 ? "" : "s"}`}
          className="evidence-frame__overlay"
          preserveAspectRatio="none"
          role="img"
          viewBox={`0 0 ${String(dimensions.width)} ${String(dimensions.height)}`}
        >
          {frame.regions.map((region, index) => {
            const rect = regionRect(region, dimensions);
            if (!rect) return null;
            const text = regionLabel(region, index);
            const fontSize = Math.max(14, dimensions.width * 0.022);
            const labelY =
              rect.y > fontSize + 8 ? rect.y - 5 : rect.y + fontSize + 5;
            const labelWidth = Math.min(
              rect.width,
              Math.max(fontSize * 3, text.length * fontSize * 0.62 + 10),
            );
            return (
              <g key={`${text}-${String(index)}`}>
                <rect className="evidence-frame__region-fill" {...rect} />
                <rect className="evidence-frame__region-line" {...rect} />
                <rect
                  className="evidence-frame__region-label"
                  height={fontSize + 5}
                  rx={4}
                  width={labelWidth}
                  x={rect.x}
                  y={labelY - fontSize - 2}
                />
                <text fontSize={fontSize} x={rect.x + 5} y={labelY}>
                  {text}
                </text>
              </g>
            );
          })}
        </svg>
      ) : null}
      {shouldDrawRegions && frame.regions.length === 0 ? (
        <div className="evidence-frame__frame-level" aria-label="Localized evidence region unavailable">
          <span>{frameLevelLabel}</span>
        </div>
      ) : null}
    </div>
  );
}

function VisualAttributionViewer({
  frame,
  label,
  onGenerate,
  requestError,
  requestState,
}: {
  frame: PublicEvidenceFrame;
  label: string;
  onGenerate?: () => void;
  requestError?: string;
  requestState: ViewerXAIState;
}) {
  const xai = frame.xai ?? null;
  const ready = requestState === "ready" || requestState === "available";
  const maps = ready ? xai?.attribution_maps ?? [] : [];
  const wholeMap = maps.find((item) => item.scope === "observation") ?? maps[0];
  const [selectedMapId, setSelectedMapId] = useState(wholeMap?.map_id);
  const [view, setView] = useState<"original" | "xai">(
    safeImageSource(wholeMap?.heatmap_image ?? null) ? "xai" : "original",
  );

  useEffect(() => {
    setSelectedMapId(wholeMap?.map_id);
    setView(safeImageSource(wholeMap?.heatmap_image ?? null) ? "xai" : "original");
  }, [frame.frame_id, wholeMap?.map_id, wholeMap?.heatmap_image]);

  const selectedMap =
    maps.find((item) => item.map_id === selectedMapId) ?? wholeMap;
  const selectedHeatmap = safeImageSource(selectedMap?.heatmap_image ?? null);
  const xaiAvailable = ready && maps.length > 0;
  const description =
    selectedMap?.scope === "phrase"
      ? `Regions whose removal most reduced model support for the phrase “${selectedMap.label}”.`
      : "Regions whose removal most reduced support for this generated observation.";
  const disclaimer =
    xai?.disclaimer ??
    "This is a post-hoc perturbation attribution of the Visual Observer. It does not affect the authoritative verification verdict.";
  const boundary =
    xai?.scientific_boundary ??
    "Supplemental visual XAI is explanatory only and does not participate in the Frozen G1 verdict.";

  return (
    <section className="visual-xai" aria-label={`XAI attribution for ${label}`}>
      <div className="visual-xai__topline">
        <div>
          <p>Visual observer explanation</p>
          <h4>Occlusion attribution</h4>
        </div>
        <span className="visual-xai__method">
          {xai ? xaiMethodLabel(xai.method) : "XAI unavailable"}
        </span>
      </div>

      {xai ? (
        <p className="visual-xai__profile">
          {xai.profile === "public" ? "Public" : "Research"} {String(xai.grid_rows)}×{String(xai.grid_columns)} attribution
          {xai.cache_hit ? " · cached" : ""}
        </p>
      ) : null}

      {xaiAvailable ? (
        <>
          <div className="visual-xai__controls">
            <div aria-label="Attribution view" className="visual-xai__chips" role="group">
              {maps.map((item) => (
                <button
                  aria-pressed={item.map_id === selectedMap?.map_id}
                  className={item.map_id === selectedMap?.map_id ? "is-active" : ""}
                  key={item.map_id}
                  onClick={() => {
                    setSelectedMapId(item.map_id);
                    if (safeImageSource(item.heatmap_image)) setView("xai");
                  }}
                  type="button"
                >
                  {item.label}
                </button>
              ))}
            </div>
            <div aria-label="Image presentation" className="visual-xai__toggle" role="group">
              <button aria-pressed={view === "original"} onClick={() => setView("original")} type="button">
                Original
              </button>
              <button
                aria-pressed={view === "xai"}
                disabled={!selectedHeatmap}
                onClick={() => setView("xai")}
                type="button"
              >
                XAI
              </button>
            </div>
          </div>

          <figure className="visual-xai__preview">
            <AnnotatedFrame
              annotated={false}
              frame={frame}
              label={label}
              sourceOverride={view === "xai" ? selectedHeatmap : undefined}
            />
            <figcaption>
              <strong>{view === "xai" ? selectedMap?.label : "Original observer source frame"}</strong>
              <span>{view === "xai" ? description : "The unmodified frame supplied to the Visual Observer."}</span>
            </figcaption>
          </figure>
          {!selectedHeatmap && view === "original" ? (
            <p className="visual-xai__unavailable" role="status">
              Attribution metadata is present, but the public heatmap image was omitted by payload safety limits.
            </p>
          ) : null}
        </>
      ) : requestState === "not_requested" ? (
        <div className="visual-xai__lazy" role="status">
          <strong>High-cost post-hoc attribution is available.</strong>
          <span>Generate model-derived occlusion attribution for this observation.</span>
          <button onClick={onGenerate} type="button">
            Generate XAI
          </button>
        </div>
      ) : requestState === "pending" ? (
        <div className="visual-xai__lazy visual-xai__lazy--pending" role="status">
          <span aria-hidden="true" className="visual-xai__spinner" />
          <strong>Generating Qwen occlusion attribution…</strong>
          <span>The authoritative verdict is already available.</span>
        </div>
      ) : (
        <div className="visual-xai__unavailable" role="status">
          <strong>Visual XAI unavailable</strong>
          <span>
            {requestError
              ? requestError
              : xai?.unavailable_reason
              ? `Safe reason: ${xai.unavailable_reason.replaceAll("_", " ")}.`
              : "This older result does not contain an XAI artifact."}
          </span>
          <span>The authoritative verification result is unaffected.</span>
        </div>
      )}

      <div className="visual-xai__disclosure">
        <p>{disclaimer}</p>
        <strong>{boundary}</strong>
      </div>
    </section>
  );
}

function EvidenceLightbox({
  frame,
  frameIndex,
  onClose,
  onGenerate,
  requestError,
  requestState,
  sourceType,
}: {
  frame: PublicEvidenceFrame;
  frameIndex: number;
  onClose: () => void;
  onGenerate?: () => void;
  requestError?: string;
  requestState: ViewerXAIState;
  sourceType: EvidenceSourceType;
}) {
  const [closing, setClosing] = useState(false);
  const closeButton = useRef<HTMLButtonElement>(null);
  const closeTimer = useRef<number | undefined>(undefined);
  const closingRef = useRef(false);
  const priorFocus = useRef<HTMLElement | null>(null);
  const label = frameName(frame, frameIndex);
  const headingId = `evidence-viewer-${String(frameIndex)}`;

  const requestClose = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    setClosing(true);
    closeTimer.current = window.setTimeout(onClose, 180);
  }, [onClose]);

  useEffect(() => {
    priorFocus.current = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButton.current?.focus();
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") requestClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      if (closeTimer.current !== undefined) window.clearTimeout(closeTimer.current);
      priorFocus.current?.focus();
    };
  }, [requestClose]);

  return createPortal(
    <div
      aria-labelledby={headingId}
      aria-modal="true"
      className={`evidence-lightbox${closing ? " evidence-lightbox--closing" : ""}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) requestClose();
      }}
      role="dialog"
    >
      <article className="evidence-lightbox__panel">
        <header>
          <div>
            <p>Grounded evidence inspection</p>
            <h3 id={headingId}>{label}</h3>
          </div>
          <button aria-label="Close evidence viewer" onClick={requestClose} ref={closeButton} type="button">
            ×
          </button>
        </header>

        {sourceType === "visual_observation" ? (
          <VisualAttributionViewer
            frame={frame}
            label={label}
            onGenerate={onGenerate}
            requestError={requestError}
            requestState={requestState}
          />
        ) : (
          <>
            <div className="evidence-lightbox__views">
              <figure>
                <AnnotatedFrame annotated={false} frame={frame} label={label} />
                <figcaption>Original public evidence frame</figcaption>
              </figure>
              <figure>
                <AnnotatedFrame annotated frame={frame} label={label} />
                <figcaption>
                  {frame.regions.length > 0
                    ? "Annotated OCR regions"
                    : "Frame-level grounding; localized region unavailable"}
                </figcaption>
              </figure>
            </div>

            <div className="evidence-lightbox__details">
              <p>{frame.explanation}</p>
              {frame.regions.length > 0 ? (
                <ul aria-label="Recognized OCR regions">
                  {frame.regions.map((region, index) => (
                    <li key={`${regionLabel(region, index)}-${String(index)}`}>
                      <strong>{regionLabel(region, index)}</strong>
                      <span>
                        {region.confidence === null
                          ? "Confidence unavailable"
                          : `Confidence ${String(region.confidence)}`}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </>
        )}
      </article>
    </div>,
    document.body,
  );
}

export function EvidenceFrameGallery({
  frames,
  jobId,
  onVisualXAIReady,
  sourceType,
  unitId,
}: EvidenceFrameGalleryProps) {
  const [selectedIndex, setSelectedIndex] = useState<number>();
  const [previewIndex, setPreviewIndex] = useState(0);
  const backendState = frames.find((frame) => frame.xai)?.xai?.status;
  const [requestState, setRequestState] = useState<ViewerXAIState>(
    backendState ?? "legacy_missing",
  );
  const [requestError, setRequestError] = useState<string>();
  const heading = sourceType === "ocr" ? "OCR Evidence Frames" : "Visual Evidence Frames";
  const closeLightbox = useCallback(() => setSelectedIndex(undefined), []);

  useEffect(() => {
    setRequestState(backendState ?? "legacy_missing");
    if (backendState === "ready" || backendState === "available") {
      setRequestError(undefined);
    }
  }, [backendState]);

  const generateVisualXAI = useCallback(() => {
    if (!jobId || !unitId || requestState === "pending") return;
    setRequestError(undefined);
    setRequestState("pending");
    void apiClient
      .requestVisualXAI(jobId, unitId)
      .then((response) => {
        const nextState = response.visual_xai.status;
        setRequestState(nextState);
        if (nextState === "ready") onVisualXAIReady?.();
      })
      .catch(() => {
        setRequestState("failed");
        setRequestError(
          "XAI attribution unavailable. The authoritative verification result is unaffected.",
        );
      });
  }, [jobId, onVisualXAIReady, requestState, unitId]);

  useEffect(() => {
    if (requestState !== "pending" || !jobId || !unitId) return undefined;
    let disposed = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const response = await apiClient.getVisualXAI(jobId, unitId);
        if (disposed) return;
        const nextState = response.visual_xai.status;
        setRequestState(nextState);
        if (nextState === "ready") {
          onVisualXAIReady?.();
          return;
        }
        if (
          nextState === "failed" ||
          nextState === "unavailable" ||
          nextState === "not_requested"
        ) {
          return;
        }
        timer = window.setTimeout(
          () => void poll(),
          response.poll_after_ms ?? 1500,
        );
      } catch {
        if (disposed) return;
        setRequestState("failed");
        setRequestError(
          "XAI attribution unavailable. The authoritative verification result is unaffected.",
        );
      }
    };

    timer = window.setTimeout(() => void poll(), 1500);
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [jobId, onVisualXAIReady, requestState, unitId]);

  return (
    <section aria-label={heading} className="evidence-frames">
      <div className="evidence-frames__heading">
        <div>
          <p>Visual evidence</p>
          <h4>{heading}</h4>
        </div>
        <span>{frames.length > 0 ? `${String(frames.length)} referenced` : "Not provided"}</span>
      </div>

      {frames.length > 0 ? (
        <>
          {sourceType === "visual_observation" && frames[previewIndex] ? (
            <VisualAttributionViewer
              frame={frames[previewIndex]}
              label={frameName(frames[previewIndex], previewIndex)}
              onGenerate={generateVisualXAI}
              requestError={requestError}
              requestState={requestState}
            />
          ) : null}
          <div className="evidence-frames__rail">
            {frames.map((frame, index) => {
              const label = frameName(frame, index);
              const attribution = defaultAttributionMap(frame);
              return (
                <button
                  aria-label={`Inspect ${label}`}
                  className={`evidence-frame-card${previewIndex === index ? " is-current" : ""}`}
                  key={`${label}-${String(index)}`}
                  onClick={() => {
                    setPreviewIndex(index);
                    setSelectedIndex(index);
                  }}
                  type="button"
                >
                  <AnnotatedFrame
                    annotated={sourceType === "ocr"}
                    frame={frame}
                    label={label}
                    sourceOverride={
                      sourceType === "visual_observation"
                        ? safeImageSource(attribution?.heatmap_image ?? null)
                        : undefined
                    }
                  />
                  <span className="evidence-frame-card__caption">
                    <strong>{label}</strong>
                    <small>{frameTime(frame) ?? "Timestamp unavailable"}</small>
                    {sourceType === "visual_observation" ? <em>Observer source</em> : null}
                  </span>
                </button>
              );
            })}
          </div>
        </>
      ) : (
        <div className="evidence-frames__empty">
          <strong>Frame imagery was not included in this public result.</strong>
          <span>No coordinates or visual regions are inferred by the interface.</span>
        </div>
      )}

      {selectedIndex !== undefined && frames[selectedIndex] ? (
        <EvidenceLightbox
          frame={frames[selectedIndex]}
          frameIndex={selectedIndex}
          onClose={closeLightbox}
          onGenerate={generateVisualXAI}
          requestError={requestError}
          requestState={requestState}
          sourceType={sourceType}
        />
      ) : null}
    </section>
  );
}
