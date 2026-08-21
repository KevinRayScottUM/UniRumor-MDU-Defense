import type { ReactNode } from "react";

import { Card } from "./Card";

export interface EmptyStateProps {
  eyebrow?: string;
  title: string;
  description: string;
  action?: ReactNode;
  headingLevel?: 2 | 3;
}

export function EmptyState({
  action,
  description,
  eyebrow,
  headingLevel = 2,
  title,
}: EmptyStateProps) {
  const Heading = `h${headingLevel}` as "h2" | "h3";

  return (
    <Card className="ui-empty-state" variant="subtle">
      <span aria-hidden="true" className="ui-empty-state__mark" />
      <div>
        {eyebrow ? <p className="ui-empty-state__eyebrow">{eyebrow}</p> : null}
        <Heading className="ui-empty-state__title">{title}</Heading>
        <p className="ui-empty-state__description">{description}</p>
      </div>
      {action ? <div className="ui-empty-state__action">{action}</div> : null}
    </Card>
  );
}
