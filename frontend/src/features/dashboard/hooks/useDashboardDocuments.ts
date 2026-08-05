import { useQuery } from '@tanstack/react-query'

import { listDocuments } from '@/services/documents.service'
import type { DocumentStats } from '@/types'
import { computeDocumentStats, sortByCreatedAtDesc } from '@/utils'

const DASHBOARD_DOCUMENT_LIMIT = 100
const RECENT_UPLOADS_LIMIT = 5

export function useDashboardDocuments() {
  return useQuery({
    queryKey: ['documents', 'dashboard', DASHBOARD_DOCUMENT_LIMIT],
    queryFn: () => listDocuments({ limit: DASHBOARD_DOCUMENT_LIMIT, offset: 0 }),
    select: (response) => {
      const sorted = sortByCreatedAtDesc(response.items)
      const stats: DocumentStats = computeDocumentStats(response.items, response.total)

      return {
        stats,
        recentUploads: sorted.slice(0, RECENT_UPLOADS_LIMIT),
        total: response.total,
      }
    },
  })
}
