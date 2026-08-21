const INVALID_API_BASE_URL_MESSAGE =
  "VITE_API_BASE_URL must be an absolute HTTP(S) URL without credentials, a query, or a fragment.";

export function resolveApiBaseUrl(value: string | undefined): string {
  const configuredValue = value?.trim() ?? "";

  // An empty value deliberately keeps development and same-origin deployments
  // local-friendly without embedding a deployment domain in the application.
  if (!configuredValue) {
    return "";
  }

  let parsed: URL;
  try {
    parsed = new URL(configuredValue);
  } catch {
    throw new TypeError(INVALID_API_BASE_URL_MESSAGE);
  }

  if (
    (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new TypeError(INVALID_API_BASE_URL_MESSAGE);
  }

  const pathPrefix = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.origin}${pathPrefix}`;
}

export const apiBaseUrl = resolveApiBaseUrl(
  import.meta.env.VITE_API_BASE_URL,
);
