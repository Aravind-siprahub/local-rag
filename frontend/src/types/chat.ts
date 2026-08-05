export interface ChatSession {
  id: string
  user_id: string
  title: string
  is_archived: boolean
  last_message_at: string | null
  created_at: string
  updated_at: string
}

export type ChatSessionListResponse = import('./api').PaginatedResponse<ChatSession>
