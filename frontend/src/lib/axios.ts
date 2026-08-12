import axios, { AxiosError } from 'axios'
import { AUTH_KEYS } from '@/features/auth/utils/constants'

// Prefer relative `/api` so the Vite dev proxy handles backend routing (no CORS).
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
})

// Request Interceptor: Attach Access Token to Requests
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem(AUTH_KEYS.ACCESS_TOKEN)
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response Interceptor: Handle Global HTTP Errors & 401 Unauthenticated
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ message?: string; detail?: string }>) => {
    if (error.response?.status === 401) {
      // Clear token on 401 unauthenticated
      localStorage.removeItem(AUTH_KEYS.ACCESS_TOKEN)
      localStorage.removeItem(AUTH_KEYS.USER_DATA)
      
      // Dispatch custom auth expired event
      window.dispatchEvent(new Event('auth:unauthorized'))
    }
    return Promise.reject(error)
  }
)

export default apiClient
