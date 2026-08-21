import type { HTMLAttributes } from "react";

import { mergeClassNames } from "./utils";

export type CardVariant = "default" | "subtle" | "raised" | "glass";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant;
}
export function Card({ className, variant = "default", ...props }: CardProps) {
  return (
    <div
      {...props}
      className={mergeClassNames("ui-card", `ui-card--${variant}`, className)}
    />
  );
}
