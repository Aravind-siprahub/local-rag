export interface PaginatedResponse<T> {
  items: T[]
  total: number
  limit: number
  offset: number
}

export interface ApiErrorBody {
  detail?: string | Record<string, unknown>
  message?: string
}

export interface PaginationParams {
  limit?: number
  offset?: number
}
