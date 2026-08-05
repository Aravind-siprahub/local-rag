import axios, { type AxiosError } from 'axios'

import type { ApiErrorBody } from '@/types'

const baseURL = import.meta.env.VITE_API_BASE_URL ?? '/api'

export const apiClient = axios.create({
  baseURL,
  headers: {
    Accept: 'application/json',
  },
  timeout: 120_000,
})

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
