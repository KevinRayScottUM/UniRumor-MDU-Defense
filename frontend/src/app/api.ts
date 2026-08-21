import { createApiClient } from "../api";
import { apiBaseUrl } from "./environment";

export const apiClient = createApiClient({
  baseUrl: apiBaseUrl,
});
