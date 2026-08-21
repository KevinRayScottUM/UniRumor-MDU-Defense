import {
  forwardRef,
  useId,
  type InputHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

import { mergeClassNames } from "./utils";

interface FieldPresentationProps {
  label: string;
  hint?: string;
  error?: string;
  fieldClassName?: string;
}

export interface InputProps
  extends InputHTMLAttributes<HTMLInputElement>,
    FieldPresentationProps {}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  {
    "aria-describedby": ariaDescribedBy,
    className,
    error,
    fieldClassName,
    hint,
    id,
    label,
    ...props
  },
  ref,
) {
  const generatedId = useId();
  const inputId = id ?? generatedId;
  const internalDescriptionId =
    hint || error ? `${inputId}-description` : undefined;
  const descriptionId = [ariaDescribedBy, internalDescriptionId]
    .filter(Boolean)
    .join(" ") || undefined;

  return (
    <div className={mergeClassNames("ui-field", fieldClassName)}>
      <label className="ui-field__label" htmlFor={inputId}>
        {label}
      </label>
      <input
        {...props}
        ref={ref}
        aria-describedby={descriptionId}
        aria-invalid={Boolean(error) || undefined}
        className={mergeClassNames("ui-input", error && "ui-input--error", className)}
        id={inputId}
      />
      {hint || error ? (
        <p
          className={mergeClassNames(
            "ui-field__description",
            error && "ui-field__description--error",
          )}
          id={internalDescriptionId}
        >
          {error ?? hint}
        </p>
      ) : null}
    </div>
  );
});

export interface TextAreaProps
  extends TextareaHTMLAttributes<HTMLTextAreaElement>,
    FieldPresentationProps {}

export const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(
  function TextArea(
    {
      "aria-describedby": ariaDescribedBy,
      className,
      error,
      fieldClassName,
      hint,
      id,
      label,
      ...props
    },
    ref,
  ) {
    const generatedId = useId();
    const inputId = id ?? generatedId;
    const internalDescriptionId =
      hint || error ? `${inputId}-description` : undefined;
    const descriptionId = [ariaDescribedBy, internalDescriptionId]
      .filter(Boolean)
      .join(" ") || undefined;

    return (
      <div className={mergeClassNames("ui-field", fieldClassName)}>
        <label className="ui-field__label" htmlFor={inputId}>
          {label}
        </label>
        <textarea
          {...props}
          ref={ref}
          aria-describedby={descriptionId}
          aria-invalid={Boolean(error) || undefined}
          className={mergeClassNames(
            "ui-input",
            "ui-textarea",
            error && "ui-input--error",
            className,
          )}
          id={inputId}
        />
        {hint || error ? (
          <p
            className={mergeClassNames(
              "ui-field__description",
              error && "ui-field__description--error",
            )}
            id={internalDescriptionId}
          >
            {error ?? hint}
          </p>
        ) : null}
      </div>
    );
  },
);
