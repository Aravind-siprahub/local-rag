import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useMemo, useState } from 'react'

import { getApiErrorMessage } from '@/api/client'
import { uploadDocument, getDocument } from '@/services'
import { getHealth } from '@/services/health.service'
import { useCurrentUser } from '@/hooks'
import type { RejectedFile, UploadQueueItem } from '@/types'

const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024 // 25 MB
const ACCEPTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md', '.markdown']

export function useUploadQueue() {
  const queryClient = useQueryClient()

  const [queue, setQueue] = useState<UploadQueueItem[]>([])
  const [rejectedFiles, setRejectedFiles] = useState<RejectedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)

  // Use resolved active current user for upload ownership
  const currentUserQuery = useCurrentUser()
  const primaryUser = currentUserQuery.data ?? null

  // Independently probe backend reachability (does not depend on users existing)
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    retry: 1,
    staleTime: 30_000,
  })
  const isBackendReachable = !healthQuery.isError

  const validateFile = useCallback((file: File, existingQueue: UploadQueueItem[]): string | null => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      return `Unsupported file format (${ext || 'unknown'}). Accepted formats: PDF, DOCX, TXT, MD.`
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      return `File exceeds maximum allowed size of 25 MB (${(file.size / (1024 * 1024)).toFixed(1)} MB).`
    }

    const isDuplicate = existingQueue.some(
      (item) => item.name === file.name && item.size === file.size,
    )
    if (isDuplicate) {
      return 'File is already in the upload queue.'
    }

    return null
  }, [])

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const fileArray = Array.from(files)
      const newItems: UploadQueueItem[] = []
      const newRejected: RejectedFile[] = []

      setQueue((prevQueue) => {
        const currentQueue = [...prevQueue]

        for (const file of fileArray) {
          const error = validateFile(file, currentQueue)
          if (error) {
            newRejected.push({ file, reason: error })
          } else {
            const newItem: UploadQueueItem = {
              id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
              file,
              name: file.name,
              size: file.size,
              type: file.type || 'application/octet-stream',
              status: 'Waiting',
              progress: 0,
            }
            newItems.push(newItem)
            currentQueue.push(newItem)
          }
        }

        return currentQueue
      })

      if (newRejected.length > 0) {
        setRejectedFiles((prev) => [...prev, ...newRejected])
      }
    },
    [validateFile],
  )

  const removeItem = useCallback((id: string) => {
    setQueue((prev) => prev.filter((item) => item.id !== id))
  }, [])

  const clearCompleted = useCallback(() => {
    setQueue((prev) => prev.filter((item) => item.status !== 'Ready' && item.status !== 'Failed'))
  }, [])

  const clearRejected = useCallback(() => {
    setRejectedFiles([])
  }, [])

  const updateItemStatus = useCallback(
    (id: string, updates: Partial<UploadQueueItem>) => {
      setQueue((prev) =>
        prev.map((item) => (item.id === id ? { ...item, ...updates } : item)),
      )
    },
    [],
  )

  const uploadSingleItem = useCallback(
    async (item: UploadQueueItem, userId: string) => {
      updateItemStatus(item.id, { status: 'Uploading', progress: 0, error: undefined })

      try {
        const response = await uploadDocument({
          file: item.file,
          userId,
          title: item.name,
          onUploadProgress: (percent) => {
            updateItemStatus(item.id, { progress: percent })
          },
        })

        updateItemStatus(item.id, {
          status: 'Parsing',
          progress: 100,
          documentId: response.document_id,
          versionId: response.version_id,
          processingJobId: response.processing_job_id,
        })

        // Poll document status instead of setTimeout
        const pollInterval = setInterval(async () => {
          try {
            const doc = await getDocument(response.document_id)
            const docStatus = typeof doc.status === 'string' ? doc.status : String(doc.status)
            if (docStatus.toLowerCase() === 'ready' || docStatus.toLowerCase() === 'failed') {
              clearInterval(pollInterval)
              updateItemStatus(item.id, {
                status: (docStatus.toLowerCase() === 'ready' ? 'Ready' : 'Failed') as any,
                error: docStatus.toLowerCase() === 'failed' ? 'Processing failed' : undefined,
              })
              void queryClient.invalidateQueries({ queryKey: ['documents'] })
            }
          } catch (e) {
            clearInterval(pollInterval)
            updateItemStatus(item.id, { status: 'Failed', error: 'Failed to poll status' })
          }
        }, 2000)

      } catch (err: unknown) {
        const errorMsg = getApiErrorMessage(err)
        updateItemStatus(item.id, {
          status: 'Failed',
          error: errorMsg,
        })
      }
    },
    [queryClient, updateItemStatus],
  )

  const uploadAll = useCallback(async () => {
    if (!primaryUser) {
      return
    }

    const pendingItems = queue.filter(
      (item) => item.status === 'Waiting' || item.status === 'Failed',
    )
    if (pendingItems.length === 0) {
      return
    }

    setIsUploading(true)

    try {
      for (const item of pendingItems) {
        await uploadSingleItem(item, primaryUser.id)
      }
    } finally {
      setIsUploading(false)
    }
  }, [primaryUser, queue, uploadSingleItem])

  const retryItem = useCallback(
    (id: string) => {
      if (!primaryUser) return
      const target = queue.find((item) => item.id === id)
      if (target) {
        void uploadSingleItem(target, primaryUser.id)
      }
    },
    [primaryUser, queue, uploadSingleItem],
  )

  const cancelItem = useCallback(
    (id: string) => {
      updateItemStatus(id, { status: 'Waiting', progress: 0, error: 'Upload cancelled.' })
    },
    [updateItemStatus],
  )

  const overallProgress = useMemo(() => {
    if (queue.length === 0) return 0
    const totalProgress = queue.reduce((acc, item) => {
      if (item.status === 'Ready' || item.status === 'Parsing' || item.status === 'Chunking' || item.status === 'Embedding') {
        return acc + 100
      }
      return acc + item.progress
    }, 0)
    return Math.round(totalProgress / queue.length)
  }, [queue])

  return {
    queue,
    rejectedFiles,
    primaryUser,
    isUserLoading: currentUserQuery.isLoading,
    isBackendReachable,
    isUploading,
    overallProgress,
    addFiles,
    removeItem,
    retryItem,
    cancelItem,
    clearCompleted,
    clearRejected,
    uploadAll,
  }
}
