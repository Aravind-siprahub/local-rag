import { apiClient } from '@/api/client'
import type { PaginationParams, User, UserListResponse } from '@/types'

export async function createUser(payload: {
  email: string
  password?: string
  full_name?: string
  role?: string
}): Promise<User> {
  // Password MUST meet the backend PasswordPolicy:
  // ≥8 chars, uppercase, lowercase, digit, special character.
  const safePassword = payload.password || 'LocalRag@Default1'
  const { data } = await apiClient.post<User>('/users', {
    email: payload.email,
    password: safePassword,
    full_name: payload.full_name || 'Default User',
    // Backend UserRole enum values are 'admin' | 'member'
    role: payload.role ?? 'member',
  })
  return data
}

export async function listUsers(params: PaginationParams = {}): Promise<UserListResponse> {
  const { data } = await apiClient.get<UserListResponse>('/users', { params })
  return data
}

export async function getUser(userId: string): Promise<User> {
  const { data } = await apiClient.get<User>(`/users/${userId}`)
  return data
}
