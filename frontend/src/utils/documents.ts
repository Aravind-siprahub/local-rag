import type {
  Document,
  DocumentDisplayStatus,
  DocumentListItem,
  DocumentVersion,
} from '@/types'

const statusKeys: import('@/types').DocumentStatus[] = [
  'uploaded',
  'processing',
  'ready',
  'failed',
  'archived',
]

export function computeDocumentStats(
  documents: Document[],
  totalFromApi: number,
): import('@/types').DocumentStats {
  const stats: import('@/types').DocumentStats = {
    total: totalFromApi,
    uploaded: 0,
    processing: 0,
    ready: 0,
    failed: 0,
    archived: 0,
  }

  for (const document of documents) {
    stats[document.status] += 1
  }

  return stats
}

export function getDocumentStatusLabel(status: import('@/types').DocumentStatus): string {
  return status.charAt(0).toUpperCase() + status.slice(1)
}

export const documentStatusOrder = statusKeys

export function getDisplayStatusLabel(status: DocumentDisplayStatus): string {
  const labels: Record<DocumentDisplayStatus, string> = {
    pending: 'Pending',
    parsing: 'Parsing',
    chunked: 'Chunked',
    embedded: 'Embedded',
    ready: 'Ready',
    failed: 'Failed',
    archived: 'Archived',
  }

  return labels[status]
}

export function resolveCurrentVersion(
  document: Document,
  versions: DocumentVersion[],
): DocumentVersion | null {
  if (versions.length === 0) {
    return null
  }

  if (document.current_version_id) {
    const current = versions.find((version) => version.id === document.current_version_id)
    if (current) {
      return current
    }
  }

  return [...versions].sort((left, right) => right.version_number - left.version_number)[0] ?? null
}

export function deriveDisplayStatus(
  document: Document,
  version: DocumentVersion | null,
): DocumentDisplayStatus {
  if (document.status === 'archived') {
    return 'archived'
  }

  if (document.status === 'failed' || version?.status === 'failed') {
    return 'failed'
  }

  if (document.status === 'ready' || version?.status === 'completed') {
    return 'ready'
  }

  if (
    version?.status === 'embedding' ||
    version?.status === 'embedded' ||
    version?.status === 'indexing'
  ) {
    return 'embedded'
  }

  if (version?.status === 'chunked') {
    return 'chunked'
  }

  if (
    document.status === 'processing' ||
    version?.status === 'parsing' ||
    version?.status === 'parsed' ||
    version?.status === 'chunking'
  ) {
    return 'parsing'
  }

  return 'pending'
}

export function buildDocumentListItem(
  document: Document,
  versions: DocumentVersion[],
): DocumentListItem {
  const currentVersion = resolveCurrentVersion(document, versions)

  return {
    document,
    filename: currentVersion?.original_filename ?? null,
    versionLabel: currentVersion ? `v${currentVersion.version_number}` : null,
    fileSizeBytes: currentVersion?.file_size_bytes ?? null,
    displayStatus: deriveDisplayStatus(document, currentVersion),
  }
}

export function filterDocumentsBySearch(
  items: DocumentListItem[],
  query: string,
): DocumentListItem[] {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) {
    return items
  }

  return items.filter((item) => {
    const title = item.document.title.toLowerCase()
    const filename = item.filename?.toLowerCase() ?? ''
    return title.includes(normalizedQuery) || filename.includes(normalizedQuery)
  })
}

export function formatFileSize(bytes: number | null | undefined): string {
  if (bytes == null || Number.isNaN(bytes)) {
    return '—'
  }

  if (bytes < 1024) {
    return `${bytes} B`
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

export function getPipelineProgress(version: DocumentVersion | null): {
  chunksCreated: boolean
  embeddingsCreated: boolean
} {
  if (!version) {
    return { chunksCreated: false, embeddingsCreated: false }
  }

  const chunkedStatuses: DocumentVersion['status'][] = [
    'chunked',
    'embedding',
    'embedded',
    'indexing',
    'completed',
  ]
  const embeddedStatuses: DocumentVersion['status'][] = ['embedded', 'indexing', 'completed']

  return {
    chunksCreated: Boolean(version.chunked_at) || chunkedStatuses.includes(version.status),
    embeddingsCreated: Boolean(version.embedded_at) || embeddedStatuses.includes(version.status),
  }
}
