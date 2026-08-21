import { TopBar } from '@/components/TopBar'
import {
  DashboardStats,
  RecentChats,
  RecentUploads,
} from '@/features/dashboard/components'
import {
  useDashboardChatSessions,
  useDashboardDocuments,
} from '@/features/dashboard/hooks'

export function DashboardPage() {
  const documentsQuery = useDashboardDocuments()
  const chatSessionsQuery = useDashboardChatSessions()

  return (
    <div className="space-y-8">
      <TopBar
        title="Knowledge Studio Dashboard"
        description="Monitor document ingestion, processing health, and recent RAG activity across your local knowledge base."
      />

      <DashboardStats stats={documentsQuery.data?.stats} isLoading={documentsQuery.isLoading} />

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="flex flex-col min-w-0">
          <RecentUploads
            documents={documentsQuery.data?.recentUploads}
            isLoading={documentsQuery.isLoading}
            isError={documentsQuery.isError}
            error={documentsQuery.error}
            onRetry={() => {
              void documentsQuery.refetch()
            }}
          />
        </div>
        
        <div className="flex flex-col min-w-0">
          <RecentChats
            sessions={chatSessionsQuery.data}
            isLoading={chatSessionsQuery.isLoading}
            isError={chatSessionsQuery.isError}
            error={chatSessionsQuery.error}
            onRetry={() => {
              void chatSessionsQuery.refetch()
            }}
          />
        </div>
      </div>
    </div>
  )
}
