import { apiClient } from '@/api/client'
import type { PaginationParams, ProcessingJobListResponse } from '@/types'

export interface ListProcessingJobsParams extends PaginationParams {
  document_version_id?: string
  active_only?: boolean
}

export async function listProcessingJobs(
  params: ListProcessingJobsParams = {},
): Promise<ProcessingJobListResponse> {
  const { data } = await apiClient.get<ProcessingJobListResponse>('/processing-jobs', { params })
  return data
}
