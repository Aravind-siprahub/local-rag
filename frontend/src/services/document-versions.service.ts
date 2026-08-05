import { apiClient } from '@/api/client'
import type { DocumentVersionListResponse, PaginationParams } from '@/types'

export interface ListDocumentVersionsParams extends PaginationParams {
  document_id: string
}

export async function listDocumentVersions(
  params: ListDocumentVersionsParams,
): Promise<DocumentVersionListResponse> {
  const { data } = await apiClient.get<DocumentVersionListResponse>('/document-versions', { params })
  return data
}

export async function getDocumentVersion(versionId: string) {
  const { data } = await apiClient.get(`/document-versions/${versionId}`)
  return data
}
