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
      if (payload.provider) {
        formData.append('provider', payload.provider)
      }
      if (payload.model) {
        formData.append('model', payload.model)
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

  /**
   * SSE Streaming alternative to sendMessage.
   * Parses backend data: {} events dynamically.
   */
  async streamMessage(
    payload: ChatRequest & { file?: File },
    callbacks: {
      onStatus?: (message: string) => void
      onToken?: (text: string) => void
      onMeta?: (sources: any[], userMessageId: string) => void
      onDone?: (data: any) => void
      onError?: (error: Error) => void
    },
    options?: { signal?: AbortSignal }
  ): Promise<void> {
    const baseURL = import.meta.env.VITE_API_BASE_URL ?? '/api'
    const url = `${baseURL}/chat/stream`
    const token = localStorage.getItem('ACCESS_TOKEN') // AUTH_KEYS.ACCESS_TOKEN

    const headers: Record<string, string> = {}
    if (token) headers['Authorization'] = `Bearer ${token}`

    const requestId = `req-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
    headers['X-Request-ID'] = requestId

    const firstImageFile =
      payload.file ||
      payload.attachments?.find((a: any) => a.file instanceof File)?.file

    let body: FormData | string
    if (firstImageFile) {
      const formData = new FormData()
      formData.append('question', payload.question)
      if (payload.session_id) formData.append('session_id', payload.session_id)
      if (payload.document_id) formData.append('document_id', payload.document_id)
      if (payload.document_version_id) formData.append('document_version_id', payload.document_version_id)
      if (payload.top_k) formData.append('top_k', String(payload.top_k))
      if (payload.similarity_threshold) formData.append('similarity_threshold', String(payload.similarity_threshold))
      if (payload.provider) formData.append('provider', payload.provider)
      if (payload.model) formData.append('model', payload.model)
      formData.append('file', firstImageFile)
      body = formData
    } else {
      headers['Content-Type'] = 'application/json'
      const jsonPayload = {
        ...payload,
        attachments: payload.attachments?.map((a: any) => ({
          id: a.id,
          filename: a.filename,
          mime_type: a.mime_type,
          size: a.size,
          document_id: a.document_id,
          storage_path: a.storage_path || a.file_path,
          url: a.url || a.previewUrl,
        })),
      }
      body = JSON.stringify(jsonPayload)
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body,
        signal: options?.signal,
      })

      if (!response.ok) {
        let errorMsg = `Server error: ${response.statusText}`
        try {
          const errBody = await response.json()
          if (errBody?.detail) {
            errorMsg = typeof errBody.detail === 'string' ? errBody.detail : JSON.stringify(errBody.detail)
          }
        } catch {
          // ignore
        }
        throw new Error(errorMsg)
      }

      if (!response.body) {
        throw new Error('ReadableStream not supported by the browser.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        let boundaryIndex
        while ((boundaryIndex = buffer.indexOf('\n\n')) >= 0) {
          const chunk = buffer.slice(0, boundaryIndex).trim()
          buffer = buffer.slice(boundaryIndex + 2)

          if (!chunk) continue

          // Handle multi-line data: events
          const lines = chunk.split('\n')
          const dataLines = lines
            .filter((l) => l.startsWith('data: '))
            .map((l) => l.substring(6))

          if (dataLines.length > 0) {
            const jsonStr = dataLines.join('')
            try {
              const event = JSON.parse(jsonStr)
              if (event.type === 'status') {
                callbacks.onStatus?.(event.message)
              } else if (event.type === 'meta') {
                callbacks.onMeta?.(event.sources, event.user_message_id)
              } else if (event.type === 'token') {
                callbacks.onToken?.(event.content)
              } else if (event.type === 'done') {
                callbacks.onDone?.(event)
              } else if (event.type === 'error') {
                throw new Error(event.error || event.message || 'Stream error')
              }
            } catch (err) {
              console.warn('[SSE PARSE ERROR] Skipping malformed chunk:', chunk, err)
            }
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        throw Object.assign(new Error('Response cancelled. Please try again.'), { code: 'TIMEOUT' })
      }
      callbacks.onError?.(err instanceof Error ? err : new Error(String(err)))
      throw err
    }
  },
}
