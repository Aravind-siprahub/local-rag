export type UserRole = 'admin' | 'hr' | 'user' | 'member'

export interface User {
  id: string
  email: string
  full_name: string | null
  role: UserRole
  is_active: boolean
  is_verified: boolean
  last_login_at: string | null
  created_at: string
  updated_at: string
}

export type UserListResponse = import('./api').PaginatedResponse<User>
