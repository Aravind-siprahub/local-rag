export type DocumentVersionStatus =
  | 'uploaded'
  | 'parsing'
  | 'parsed'
  | 'chunking'
  | 'chunked'
  | 'embedding'
  | 'embedded'
  | 'indexing'
  | 'completed'
  | 'failed'

export interface DocumentVersion {
  id: string
  document_id: string
  uploaded_by: string
  version_number: number
  storage_key: string
  original_filename: string
  mime_type: string
  file_size_bytes: number
  checksum_sha256: string
  page_count: number | null
  status: DocumentVersionStatus
  error_message: string | null
  parsed_at: string | null
  chunked_at: string | null
  embedded_at: string | null
  completed_at: string | null
  created_at: string
  updated_at: string
}

export type DocumentVersionListResponse = import('./api').PaginatedResponse<DocumentVersion>
