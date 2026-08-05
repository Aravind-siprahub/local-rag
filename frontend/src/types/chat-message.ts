export type MessageRole = 'system' | 'user' | 'assistant'

export interface ChatMessage {
  id: string
  session_id: string
  role: MessageRole
  content: string
  created_at: string
}

export type ChatMessageListResponse = import('./api').PaginatedResponse<ChatMessage>
