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

  /**
   * Chat generation can exceed the default API client timeout.
   * Use a dedicated timeout + optional AbortSignal so the UI can recover
   * without locking the rest of the app.
   */
  async sendMessage(
    payload: ChatRequest & { file?: File },
    options?: { signal?: AbortSignal; timeoutMs?: number },
  ): Promise<ChatResponse> {
    if (payload.file) {
      // Log safe metadata only — never log bytes/base64
      console.log('[IMAGE_UPLOAD] image_selected:', {
        filename: payload.file.name,
        mime_type: payload.file.type,
        size_bytes: payload.file.size,
        has_file: true,
      })

      const formData = new FormData()
      formData.append('question', payload.question)
      if (payload.session_id) {
        formData.append('session_id', payload.session_id)
      }
      if (payload.document_id) {
        formData.append('document_id', payload.document_id)
      }
      if (payload.document_version_id) {
        formData.append('document_version_id', payload.document_version_id)
      }
      if (payload.top_k) {
        formData.append('top_k', String(payload.top_k))
      }
      if (payload.similarity_threshold) {
        formData.append('similarity_threshold', String(payload.similarity_threshold))
      }
      formData.append('file', payload.file)

      console.log('[IMAGE_UPLOAD] multipart_request_sent — letting browser set Content-Type with boundary')

      // DO NOT set Content-Type manually — the browser must set it with the
      // auto-generated boundary (e.g. "multipart/form-data; boundary=----Xyz").
      // Manually setting it strips the boundary, breaking server-side parsing.
      const { data } = await apiClient.post<ChatResponse>('/chat', formData, {
        signal: options?.signal,
        timeout: options?.timeoutMs ?? 600_000,
      })
      return data
    } else {
      const { data } = await apiClient.post<ChatResponse>('/chat', payload, {
        signal: options?.signal,
        timeout: options?.timeoutMs ?? 600_000,
      })
      return data
    }
  },

}
