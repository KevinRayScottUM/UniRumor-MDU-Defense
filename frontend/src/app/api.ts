import { createApiClient } from "../api";

export const apiClient = createApiClient({
  baseUrl: import.meta.env.VITE_API_BASE_URL,
});

