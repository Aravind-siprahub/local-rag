import axios, { type AxiosError } from 'axios'

import type { ApiErrorBody } from '@/types'

const baseURL = import.meta.env.VITE_API_BASE_URL ?? '/api'

export const apiClient = axios.create({
  baseURL,
  headers: {
    Accept: 'application/json',
  },
  timeout: 600_000,
})

apiClient.interceptors.request.use(
  (config) => {
    if (!config.headers['X-Request-ID']) {
      const requestId = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `req-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
      config.headers['X-Request-ID'] = requestId
    }
    console.log(`[AXIOS REQUEST] ${config.method?.toUpperCase()} ${config.url} (Request-ID: ${config.headers['X-Request-ID']})`);
    return config;
  },
  (error) => {
    console.error("[AXIOS ERROR]", error);
    return Promise.reject(error);
  }
);

apiClient.interceptors.response.use(
  (response) => {
    console.log("[13] AXIOS RESPONSE");
    console.log(response.status);
    return response;
  },
  (error) => {
    console.error("[14] AXIOS ERROR", error);
    return Promise.reject(error);
  }
);

export function getApiErrorMessage(error: unknown): string {
  if (!axios.isAxiosError(error)) {
    return error instanceof Error ? error.message : 'An unexpected error occurred.'
  }

  const axiosError = error as AxiosError<ApiErrorBody>
  const detail = axiosError.response?.data?.detail

  if (typeof detail === 'string') {
    return detail
  }

  if (detail && typeof detail === 'object' && 'message' in detail) {
    const message = detail.message
    if (typeof message === 'string') {
      return message
    }
  }

  if (axiosError.response?.data?.message) {
    return axiosError.response.data.message
  }

  if (axiosError.code === 'ECONNABORTED') {
    return 'Request timed out. The server may still be processing.'
  }

  if (!axiosError.response) {
    return 'Unable to reach the API. Is the backend running?'
  }

  return axiosError.message || 'Request failed.'
}
