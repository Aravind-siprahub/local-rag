import { apiClient } from '@/api/client'
import type { PaginationParams, PaginatedResponse } from '@/types'
import type { ChatRequest, ChatResponse, Conversation, Message } from '../types/chat'

export const chatService = {
  async createConversation(userId: string, title?: string): Promise<Conversation> {
    const { data } = await apiClient.post<Conversation>('/chat-sessions', {
      user_id: userId,
      title: title || 'New chat',
    })
    return data
  },

  async listConversations(userId?: string, params?: PaginationParams): Promise<PaginatedResponse<Conversation>> {
    const { data } = await apiClient.get<PaginatedResponse<Conversation>>('/chat-sessions', {
      params: { ...params, user_id: userId },
    })
    return data
  },

  async getConversation(id: string): Promise<Conversation> {
    const { data } = await apiClient.get<Conversation>(`/chat-sessions/${id}`)
    return data
  },

  async deleteConversation(id: string): Promise<void> {
    await apiClient.delete(`/chat-sessions/${id}`)
  },

  async getConversationMessages(id: string, params?: PaginationParams): Promise<PaginatedResponse<Message>> {
    const { data } = await apiClient.get<PaginatedResponse<Message>>(`/chat/sessions/${id}/messages`, { params })
    return data
  },

  /**
   * Chat generation can exceed the default API client timeout.
   * Use a dedicated timeout + optional AbortSignal so the UI can recover
   * without locking the rest of the app.
   */
  async sendMessage(
    payload: ChatRequest,
    options?: { signal?: AbortSignal; timeoutMs?: number },
  ): Promise<ChatResponse> {
    const { data } = await apiClient.post<ChatResponse>('/chat', payload, {
      signal: options?.signal,
      // LLM + RAG answers often take longer than other API calls.
      timeout: options?.timeoutMs ?? 600_000,
    })
    return data
  },
}
