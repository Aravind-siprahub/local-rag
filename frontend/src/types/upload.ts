export type UploadQueueStatus =
  | 'Waiting'
  | 'Uploading'
  | 'Parsing'
  | 'Chunking'
  | 'Embedding'
  | 'Ready'
  | 'Failed'

export interface UploadQueueItem {
  id: string
  file: File
  name: string
  size: number
  type: string
  status: UploadQueueStatus
  progress: number
  error?: string
  documentId?: string
  versionId?: string
  processingJobId?: string
}

export interface RejectedFile {
  file: File
  reason: string
}

export interface DocumentUploadResponse {
  document_id: string
  version_id: string
  processing_job_id: string
  original_filename: string
  mime_type: string
  file_size_bytes: number
  checksum_sha256: string
  storage_key: string
}
