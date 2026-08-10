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
  const cleanParams: Record<string, any> = {}

  if (
    params.user_id &&
    typeof params.user_id === 'string' &&
    params.user_id.trim() !== '' &&
    params.user_id !== 'undefined' &&
    params.user_id !== 'null'
  ) {
    cleanParams.user_id = params.user_id.trim()
  }

  if (typeof params.include_archived === 'boolean') {
    cleanParams.include_archived = params.include_archived
  }

  if (typeof params.limit === 'number') {
    cleanParams.limit = params.limit
  }

  if (typeof params.offset === 'number') {
    cleanParams.offset = params.offset
  }

  const { data } = await apiClient.get<ChatSessionListResponse>('/chat-sessions', {
    params: cleanParams,
  })
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
