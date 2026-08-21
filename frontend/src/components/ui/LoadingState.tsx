import { mergeClassNames } from "./utils";

export interface LoadingStateProps {
  label?: string;
  detail?: string;
  compact?: boolean;
  className?: string;
}
export function LoadingState({
  className,
  compact = false,
  detail,
  label = "Loading",
}: LoadingStateProps) {
  return (
    <div
      aria-live="polite"
      className={mergeClassNames(
        "ui-loading-state",
        compact && "ui-loading-state--compact",
        className,
      )}
      role="status"
    >
      <span aria-hidden="true" className="ui-spinner" />
      <div>
        <p className="ui-loading-state__label">{label}</p>
        {detail ? <p className="ui-loading-state__detail">{detail}</p> : null}
      </div>
    </div>
  );
}
