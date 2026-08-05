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

  async sendMessage(payload: ChatRequest): Promise<ChatResponse> {
    const { data } = await apiClient.post<ChatResponse>('/chat', payload)
    return data
  },
}
