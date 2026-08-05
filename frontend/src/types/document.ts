export type DocumentStatus =
  | 'uploaded'
  | 'processing'
  | 'ready'
  | 'failed'
  | 'archived'

export type DocumentDisplayStatus =
  | 'pending'
  | 'parsing'
  | 'chunked'
  | 'embedded'
  | 'ready'
  | 'failed'
  | 'archived'

export interface Document {
  id: string
  user_id: string
  title: string
  description: string | null
  tags: string[]
  status: DocumentStatus
  current_version_id: string | null
  created_at: string
  updated_at: string
}

export type DocumentListResponse = import('./api').PaginatedResponse<Document>

export interface DocumentStats {
  total: number
  uploaded: number
  processing: number
  ready: number
  failed: number
  archived: number
}

export interface DocumentListItem {
  document: Document
  filename: string | null
  versionLabel: string | null
  fileSizeBytes: number | null
  displayStatus: DocumentDisplayStatus
}
