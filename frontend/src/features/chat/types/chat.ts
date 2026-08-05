export interface Citation {
  chunk_id: string
  chunk_text: string
  document_id: string
  document_version_id: string
  similarity_score: number
  rank: number
}

export interface ChatTokenUsageResponse {
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
  token_usage?: ChatTokenUsageResponse
  model: string
  processing_time_ms: number
  user_message_id: string
  assistant_message_id: string
}

export interface ChatRequest {
  session_id?: string
  question: string
  document_id?: string
  document_version_id?: string
  top_k?: number
  similarity_threshold?: number
}

export interface Conversation {
  id: string
  title: string
  user_id: string
  is_archived: boolean
  last_message_at?: string
  created_at: string
}

export interface Message {
  id: string
  session_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  model_used?: string
  prompt_tokens?: number
  completion_tokens?: number
  latency_ms?: number
  generation_time_ms?: number
  total_tokens?: number
  error_message?: string
  created_at: string
}
