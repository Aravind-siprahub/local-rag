export interface Citation {
  chunk_id: string
  chunk_text: string
  document_id: string
  document_version_id?: string
  similarity_score: number
  rank: number
  document_title?: string
  section_title?: string
  page_number?: number
  url?: string
  domain?: string
  source_type?: 'local' | 'web'
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
  retrieval_mode?: 'local' | 'web' | 'hybrid'
}

export interface ChatRequest {
  session_id?: string
  question: string
  document_id?: string
  document_version_id?: string
  top_k?: number
  similarity_threshold?: number
  attachments?: Attachment[]
  provider?: string
  model?: string
}


export interface Conversation {
  id: string
  title: string
  user_id: string
  is_archived: boolean
  last_message_at?: string
  created_at: string
}

export interface Attachment {
  id: string
  mime_type: string
  filename: string
  size: number
  timestamp?: string
  url?: string
  previewUrl?: string
  file?: File
  document_id?: string
  status?: 'uploading' | 'ready' | 'error'
  progress?: number
  error?: string
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
  ttft_ms?: number
  generation_time_ms?: number
  total_tokens?: number
  error_message?: string
  created_at: string
  attachments?: Attachment[]
  citations?: Citation[]
  localImageUrl?: string
  retrieval_mode?: 'local' | 'web' | 'hybrid'
}

