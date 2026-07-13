/**
 * Minimal typed fetch wrapper around the FastAPI backend.
 *
 * All requests are same-origin: the Vite dev server proxies ``/api`` to
 * uvicorn, and in production FastAPI serves this bundle itself.
 */

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function qs(params?: Record<string, unknown>): string {
  if (!params) return "";
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) {
      // Repeat the key per element (FastAPI list params); an empty array
      // omits the parameter entirely.
      value.forEach((entry) => search.append(key, String(entry)));
    } else {
      search.set(key, String(value));
    }
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

async function request<T>(
  method: string,
  path: string,
  options: { params?: Record<string, unknown>; body?: unknown } = {},
): Promise<T> {
  const response = await fetch(`/api${path}${qs(options.params)}`, {
    method,
    headers:
      options.body !== undefined
        ? { "Content-Type": "application/json" }
        : undefined,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string, params?: Record<string, unknown>) =>
    request<T>("GET", path, { params }),
  post: <T>(path: string, body?: unknown, params?: Record<string, unknown>) =>
    request<T>("POST", path, { body, params }),
  patch: <T>(path: string, body?: unknown, params?: Record<string, unknown>) =>
    request<T>("PATCH", path, { body, params }),
  put: <T>(path: string, body?: unknown, params?: Record<string, unknown>) =>
    request<T>("PUT", path, { body, params }),
  // A few DELETE routes take a JSON body rather than query params (FastAPI
  // allows it): ``DELETE /datasets/{id}/media`` unlinks a list of media.
  // Passing that list as query params 422s against the pydantic body.
  del: <T>(
    path: string,
    params?: Record<string, unknown>,
    body?: unknown,
  ) => request<T>("DELETE", path, { params, body }),
};
