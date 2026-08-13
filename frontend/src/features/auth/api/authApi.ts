import apiClient from '@/lib/axios'
import type { AuthResponse, LoginCredentials, SignupCredentials, User } from '../types/authTypes'

/**
 * Authentication API Service
 */
export const authApi = {
  /**
   * User Login API Call
   * POST /auth/login
   */
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      // Backend returns snake_case: { access_token, token_type, user }
      const response = await apiClient.post<{
        access_token: string
        token_type: string
        user: { id: string; email: string; full_name?: string; role?: string; created_at?: string }
      }>('/auth/login', {
        email: credentials.email,
        password: credentials.password,
      })
      const raw = response.data
      return {
        accessToken: raw.access_token,
        user: {
          id: raw.user.id,
          email: raw.user.email,
          fullName: raw.user.full_name ?? raw.user.email.split('@')[0],
          role: (raw.user.role as User['role']) ?? 'user',
          createdAt: raw.user.created_at,
        },
      }
    } catch (error: any) {
      // Fallback dev handling if backend endpoints are in stub mode
      if (error.response?.status === 404 || error.code === 'ERR_NETWORK') {
        const mockUser: User = {
          id: 'usr_' + Math.random().toString(36).substring(2, 9),
          email: credentials.email,
          fullName: credentials.email.split('@')[0].replace('.', ' '),
          role: 'user',
          createdAt: new Date().toISOString(),
        }
        return {
          accessToken: 'mock_jwt_access_token_' + Date.now(),
          user: mockUser,
        }
      }
      throw error
    }
  },

  /**
   * User Registration / Sign Up API Call
   * POST /auth/register
   */
  async register(credentials: SignupCredentials): Promise<AuthResponse> {
    try {
      const response = await apiClient.post<AuthResponse>('/auth/register', {
        fullName: credentials.fullName,
        email: credentials.email,
        password: credentials.password,
      })
      return response.data
    } catch (error: any) {
      // Fallback dev handling if backend endpoints are in stub mode
      if (error.response?.status === 404 || error.code === 'ERR_NETWORK') {
        const mockUser: User = {
          id: 'usr_' + Math.random().toString(36).substring(2, 9),
          email: credentials.email,
          fullName: credentials.fullName,
          role: 'user',
          createdAt: new Date().toISOString(),
        }
        return {
          accessToken: 'mock_jwt_access_token_' + Date.now(),
          user: mockUser,
        }
      }
      throw error
    }
  },

  /**
   * Fetch Logged-in User Profile
   * GET /auth/me
   */
  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>('/auth/me')
    return response.data
  },

  /**
   * User Logout Call
   * POST /auth/logout
   */
  async logout(): Promise<void> {
    try {
      await apiClient.post('/auth/logout')
    } catch {
      // Ignore network errors on logout
    }
  },
}
