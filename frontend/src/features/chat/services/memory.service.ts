import { apiClient } from '@/api/client'

export interface LongTermMemory {
  id: string
  user_id: string
  memory_type: string
  content: string
  importance: number
  confidence: number
  is_active: boolean
  created_at: string
  updated_at: string
  last_accessed_at?: string
  source_conversation_id?: string
  metadata?: Record<string, any>
}

export interface MemoryListResponse {
  items: LongTermMemory[]
  total: number
}

export interface MemoryCreatePayload {
  memory_type?: string
  content: string
  importance?: number
  confidence?: number
  metadata?: Record<string, any>
}

export const memoryService = {
  async listMemories(params?: {
    memory_type?: string
    is_active?: boolean
    limit?: number
    offset?: number
  }): Promise<MemoryListResponse> {
    const { data } = await apiClient.get<MemoryListResponse>('/memory', { params })
    return data
  },

  async createMemory(payload: MemoryCreatePayload): Promise<LongTermMemory> {
    const { data } = await apiClient.post<LongTermMemory>('/memory', payload)
    return data
  },

  async deleteMemory(id: string): Promise<void> {
    await apiClient.delete(`/memory/${id}`)
  },

  async purgeAllMemories(): Promise<{ deleted_count: number; message: string }> {
    const { data } = await apiClient.delete<{ deleted_count: number; message: string }>('/memory')
    return data
  },
}
