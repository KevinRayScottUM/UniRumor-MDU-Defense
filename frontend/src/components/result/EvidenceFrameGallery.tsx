import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type SyntheticEvent,
} from "react";
import { createPortal } from "react-dom";

import type {
  EvidenceSourceType,
  PublicEvidenceFrame,
  PublicEvidenceRegion,
} from "../../types";

export interface EvidenceFrameGalleryProps {
  frames: PublicEvidenceFrame[];
  sourceType: EvidenceSourceType;
}

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
}: {
  annotated: boolean;
  frame: PublicEvidenceFrame;
  label: string;
}) {
  const [dimensions, setDimensions] = useState<ImageDimensions>();
  const original = safeImageSource(frame.original_image);
  const serverAnnotated = safeImageSource(frame.annotated_image);
  const source = annotated ? serverAnnotated ?? original : original;
  const shouldDrawRegions = annotated && !serverAnnotated;
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
      <img alt={`${label}${annotated ? " annotated evidence" : " original"}`} onLoad={recordDimensions} src={source} />
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

function EvidenceLightbox({
  frame,
  frameIndex,
  onClose,
}: {
  frame: PublicEvidenceFrame;
  frameIndex: number;
  onClose: () => void;
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
      </article>
    </div>,
    document.body,
  );
}

export function EvidenceFrameGallery({ frames, sourceType }: EvidenceFrameGalleryProps) {
  const [selectedIndex, setSelectedIndex] = useState<number>();
  const heading = sourceType === "ocr" ? "OCR Evidence Frames" : "Visual Evidence Frames";
  const closeLightbox = useCallback(() => setSelectedIndex(undefined), []);

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
        <div className="evidence-frames__rail">
          {frames.map((frame, index) => {
            const label = frameName(frame, index);
            return (
              <button
                aria-label={`Inspect ${label}`}
                className="evidence-frame-card"
                key={`${label}-${String(index)}`}
                onClick={() => setSelectedIndex(index)}
                type="button"
              >
                <AnnotatedFrame annotated frame={frame} label={label} />
                <span className="evidence-frame-card__caption">
                  <strong>{label}</strong>
                  <small>{frameTime(frame) ?? "Timestamp unavailable"}</small>
                </span>
              </button>
            );
          })}
        </div>
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
        />
      ) : null}
    </section>
  );
}
