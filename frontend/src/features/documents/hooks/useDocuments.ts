import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'

import { listDocumentVersions } from '@/services/document-versions.service'
import { deleteDocument, getDocument, listDocuments } from '@/services/documents.service'
import { listProcessingJobs } from '@/services/processing-jobs.service'
import type { DocumentListItem, DocumentVersion } from '@/types'
import { buildDocumentListItem, resolveCurrentVersion, sortByCreatedAtDesc } from '@/utils'

const DOCUMENTS_QUERY_KEY = ['documents', 'list'] as const
const DOCUMENTS_LIST_LIMIT = 100

export function useDocumentsList() {
  const documentsQuery = useQuery({
    queryKey: [...DOCUMENTS_QUERY_KEY, DOCUMENTS_LIST_LIMIT],
    queryFn: () => listDocuments({ limit: DOCUMENTS_LIST_LIMIT, offset: 0 }),
  })

  const documentIds = useMemo(
    () => documentsQuery.data?.items.map((document) => document.id) ?? [],
    [documentsQuery.data?.items],
  )

  const versionQueries = useQueries({
    queries: documentIds.map((documentId) => ({
      queryKey: ['document-versions', documentId],
      queryFn: () => listDocumentVersions({ document_id: documentId, limit: 50, offset: 0 }),
      enabled: Boolean(documentId),
    })),
  })

  const items = useMemo<DocumentListItem[]>(() => {
    if (!documentsQuery.data) {
      return []
    }

    const versionsByDocumentId = new Map<string, DocumentVersion[]>(
      documentIds.map((documentId, index) => [documentId, versionQueries[index]?.data?.items ?? []]),
    )

    return sortByCreatedAtDesc(documentsQuery.data.items).map((document) =>
      buildDocumentListItem(document, versionsByDocumentId.get(document.id) ?? []),
    )
  }, [documentIds, documentsQuery.data, versionQueries])

  const isLoadingVersions = versionQueries.some((query) => query.isLoading)
  const isErrorVersions = versionQueries.some((query) => query.isError)
  const versionError = versionQueries.find((query) => query.error)?.error

  return {
    items,
    total: documentsQuery.data?.total ?? 0,
    isLoading: documentsQuery.isLoading || isLoadingVersions,
    isError: documentsQuery.isError || isErrorVersions,
    error: documentsQuery.error ?? versionError,
    refetch: async () => {
      await documentsQuery.refetch()
      await Promise.all(versionQueries.map((query) => query.refetch()))
    },
  }
}

export function useDeleteDocument() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: deleteDocument,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEY })
      await queryClient.invalidateQueries({ queryKey: ['document-versions'] })
    },
  })
}

export function useDocumentDetail(documentId: string | null) {
  const documentQuery = useQuery({
    queryKey: ['documents', 'detail', documentId],
    queryFn: () => getDocument(documentId!),
    enabled: Boolean(documentId),
  })

  const versionsQuery = useQuery({
    queryKey: ['document-versions', 'detail', documentId],
    queryFn: () => listDocumentVersions({ document_id: documentId!, limit: 50, offset: 0 }),
    enabled: Boolean(documentId),
  })

  const currentVersion = useMemo(() => {
    if (!documentQuery.data || !versionsQuery.data) {
      return null
    }

    return resolveCurrentVersion(documentQuery.data, versionsQuery.data.items)
  }, [documentQuery.data, versionsQuery.data])

  const processingJobsQuery = useQuery({
    queryKey: ['processing-jobs', currentVersion?.id],
    queryFn: () =>
      listProcessingJobs({
        document_version_id: currentVersion!.id,
        limit: 50,
        offset: 0,
      }),
    enabled: Boolean(currentVersion?.id),
  })

  const listItem = useMemo(() => {
    if (!documentQuery.data || !versionsQuery.data) {
      return null
    }

    return buildDocumentListItem(documentQuery.data, versionsQuery.data.items)
  }, [documentQuery.data, versionsQuery.data])

  return {
    document: documentQuery.data ?? null,
    versions: versionsQuery.data?.items ?? [],
    currentVersion,
    processingJobs: processingJobsQuery.data?.items ?? [],
    listItem,
    isLoading:
      documentQuery.isLoading ||
      versionsQuery.isLoading ||
      (Boolean(currentVersion?.id) && processingJobsQuery.isLoading),
    isError: documentQuery.isError || versionsQuery.isError || processingJobsQuery.isError,
    error: documentQuery.error ?? versionsQuery.error ?? processingJobsQuery.error,
    refetch: async () => {
      await Promise.all([
        documentQuery.refetch(),
        versionsQuery.refetch(),
        processingJobsQuery.refetch(),
      ])
    },
  }
}
