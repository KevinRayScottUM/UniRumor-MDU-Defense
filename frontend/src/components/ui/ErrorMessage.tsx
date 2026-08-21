import { mergeClassNames } from "./utils";

export interface ErrorMessageProps {
  title?: string;
  message: string;
  requestId?: string;
  className?: string;
}
export function ErrorMessage({
  className,
  message,
  requestId,
  title = "Something went wrong",
}: ErrorMessageProps) {
  return (
    <div className={mergeClassNames("ui-error-message", className)} role="alert">
      <span aria-hidden="true" className="ui-error-message__mark">
        !
      </span>
      <div>
        <h2 className="ui-error-message__title">{title}</h2>
        <p className="ui-error-message__body">{message}</p>
        {requestId ? (
          <p className="ui-error-message__request-id">Request ID: {requestId}</p>
        ) : null}
      </div>
    </div>
  );
}
