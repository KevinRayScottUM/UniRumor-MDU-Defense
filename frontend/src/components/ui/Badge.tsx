import type { HTMLAttributes } from "react";

import { mergeClassNames } from "./utils";

export type BadgeTone =
  | "neutral"
  | "info"
  | "success"
  | "warning"
  | "danger"
  | "research";

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  withDot?: boolean;
}
export function Badge({
  children,
  className,
  tone = "neutral",
  withDot = false,
  ...props
}: BadgeProps) {
  return (
    <span
      {...props}
      className={mergeClassNames("ui-badge", `ui-badge--${tone}`, className)}
    >
      {withDot ? <span aria-hidden="true" className="ui-badge__dot" /> : null}
      {children}
    </span>
  );
}
