import { apiClient } from '@/api/client'
import type { Document, DocumentListResponse, PaginationParams } from '@/types'

export interface ListDocumentsParams extends PaginationParams {
  user_id?: string
}

export async function listDocuments(
  params: ListDocumentsParams = {},
): Promise<DocumentListResponse> {
  const { data } = await apiClient.get<DocumentListResponse>('/documents', { params })
  return data
}

export async function getDocument(documentId: string): Promise<Document> {
  const { data } = await apiClient.get<Document>(`/documents/${documentId}`)
  return data
}

export async function deleteDocument(documentId: string): Promise<void> {
  await apiClient.delete(`/documents/${documentId}`)
}
