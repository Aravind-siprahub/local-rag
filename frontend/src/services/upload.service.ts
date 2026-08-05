import { apiClient } from '@/api/client'
import type { DocumentUploadResponse } from '@/types'

export interface UploadDocumentParams {
  file: File
  userId: string
  title?: string
  onUploadProgress?: (progressPercentage: number) => void
}

export async function uploadDocument({
  file,
  userId,
  title,
  onUploadProgress,
}: UploadDocumentParams): Promise<DocumentUploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('user_id', userId)
  if (title?.trim()) {
    formData.append('title', title.trim())
  }

  const { data } = await apiClient.post<DocumentUploadResponse>('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && onUploadProgress) {
        const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        onUploadProgress(percentage)
      }
    },
  })

  return data
}
