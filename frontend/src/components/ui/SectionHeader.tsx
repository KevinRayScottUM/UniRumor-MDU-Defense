import type { ReactNode } from "react";

import { mergeClassNames } from "./utils";

export interface SectionHeaderProps {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
  headingLevel?: 1 | 2 | 3;
  headingId?: string;
  className?: string;
}

export function SectionHeader({
  action,
  className,
  description,
  eyebrow,
  headingId,
  headingLevel = 2,
  title,
}: SectionHeaderProps) {
  const Heading = `h${headingLevel}` as "h1" | "h2" | "h3";

  return (
    <div className={mergeClassNames("ui-section-header", className)}>
      <div className="ui-section-header__copy">
        {eyebrow ? <p className="ui-section-header__eyebrow">{eyebrow}</p> : null}
        <Heading className="ui-section-header__title" id={headingId}>
          {title}
        </Heading>
        {description ? (
          <p className="ui-section-header__description">{description}</p>
        ) : null}
      </div>
      {action ? <div className="ui-section-header__action">{action}</div> : null}
    </div>
  );
}
