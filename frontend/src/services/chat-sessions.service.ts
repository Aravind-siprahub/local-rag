import { apiClient } from '@/api/client'
import type { ChatMessageListResponse } from '@/types/chat-message'
import type { ChatSession, ChatSessionListResponse, PaginationParams } from '@/types'

export interface ListChatSessionsParams extends PaginationParams {
  user_id?: string
  include_archived?: boolean
}

export async function listChatSessions(
  params: ListChatSessionsParams = {},
): Promise<ChatSessionListResponse> {
  const { data } = await apiClient.get<ChatSessionListResponse>('/chat-sessions', { params })
  return data
}

export async function createChatSession(payload: {
  user_id: string
  title?: string
}): Promise<ChatSession> {
  const { data } = await apiClient.post<ChatSession>('/chat-sessions', payload)
  return data
}

export async function getChatSessionMessages(
  sessionId: string,
  params: PaginationParams = {},
): Promise<ChatMessageListResponse> {
  const { data } = await apiClient.get<ChatMessageListResponse>(
    `/chat-sessions/${sessionId}/messages`,
    { params },
  )
  return data
}
