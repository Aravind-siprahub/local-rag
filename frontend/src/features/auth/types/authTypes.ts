/**
 * Authentication Data Models & API Interfaces for "Talk to My Data"
 */

export interface User {
  id: string
  email: string
  fullName: string
  avatarUrl?: string
  role?: 'user' | 'admin'
  createdAt?: string
}

export interface LoginCredentials {
  email: string
  password: string
  rememberMe?: boolean
}

export interface SignupCredentials {
  fullName: string
  email: string
  password: string
  confirmPassword: string
  acceptTerms: boolean
}

export interface AuthResponse {
  accessToken: string
  refreshToken?: string
  user: User
  expiresIn?: number
}

export interface ApiErrorResponse {
  message: string
  code?: string
  errors?: Record<string, string[]>
}

export interface AuthState {
  user: User | null
  accessToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
}
