import { apiClient } from '@/api/client'
import type { PaginationParams, PaginatedResponse } from '@/types'
import type { ChatRequest, ChatResponse, Conversation, Message } from '../types/chat'

export const chatService = {
  async createConversation(userId?: string, title?: string): Promise<Conversation> {
    const payload: Record<string, any> = {
      title: title || 'New chat',
    }
    if (userId && typeof userId === 'string' && userId.trim() !== '' && userId !== 'undefined') {
      payload.user_id = userId.trim()
    }
    const { data } = await apiClient.post<Conversation>('/chat-sessions', payload)
    return data
  },


  async listConversations(userId?: string, params?: PaginationParams): Promise<PaginatedResponse<Conversation>> {
    const cleanParams: Record<string, any> = { ...params }
    if (
      userId &&
      typeof userId === 'string' &&
      userId.trim() !== '' &&
      userId !== 'undefined' &&
      userId !== 'null'
    ) {
      cleanParams.user_id = userId.trim()
    } else {
      delete cleanParams.user_id
    }

    const { data } = await apiClient.get<PaginatedResponse<Conversation>>('/chat-sessions', {
      params: cleanParams,
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
    console.log("[11] chatService.sendMessage()");
    console.log(payload);
    const startMono = performance.now()
    console.log('[FRONTEND REQUEST STARTED]', { question: payload.question, session_id: payload.session_id, timestamp: new Date().toISOString() })
    try {
      const { data } = await apiClient.post<ChatResponse>('/chat', payload)
      const elapsedMs = Math.round(performance.now() - startMono)
      console.log('[FRONTEND REQUEST ENDED]', { elapsed_ms: elapsedMs, assistant_message_id: data.assistant_message_id })
      return data
    } catch (error) {
      const elapsedMs = Math.round(performance.now() - startMono)
      console.error('[FRONTEND REQUEST FAILED]', { elapsed_ms: elapsedMs, error })
      throw error
    }
  },
}
