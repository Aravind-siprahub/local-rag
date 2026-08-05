export type ProcessingJobType = 'upload' | 'parse' | 'chunk' | 'embed' | 'index'

export type ProcessingJobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface ProcessingJob {
  id: string
  document_version_id: string
  job_type: ProcessingJobType
  status: ProcessingJobStatus
  error_message: string | null
  retry_count: number
  started_at: string | null
  completed_at: string | null
  created_at: string
  updated_at: string
}

export type ProcessingJobListResponse = import('./api').PaginatedResponse<ProcessingJob>
