import { apiClient } from '@/api/client'
import type { PaginationParams, User, UserListResponse } from '@/types'

export async function listUsers(params: PaginationParams = {}): Promise<UserListResponse> {
  const { data } = await apiClient.get<UserListResponse>('/users', { params })
  return data
}

export async function getUser(userId: string): Promise<User> {
  const { data } = await apiClient.get<User>(`/users/${userId}`)
  return data
}
