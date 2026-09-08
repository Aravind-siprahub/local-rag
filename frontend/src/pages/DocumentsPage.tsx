import { FileTextIcon } from 'lucide-react'
import { useMemo, useState } from 'react'

import { EmptyState } from '@/components/EmptyState'
import { ErrorState } from '@/components/ErrorState'
import { LoadingState } from '@/components/LoadingState'
import { TopBar } from '@/components/TopBar'
import {
  DeleteDocumentDialog,
  DocumentDetailDrawer,
  DocumentsCardList,
  DocumentsTable,
  DocumentsToolbar,
} from '@/features/documents/components'
import { useDocumentsList } from '@/features/documents/hooks'
import { useAuth } from '@/features/auth/hooks/authHooks'
import type { DocumentListItem } from '@/types'
import { filterDocumentsBySearch } from '@/utils'

export function DocumentsPage() {
  const { user } = useAuth()
  const isAdmin = user?.role?.toLowerCase() === 'admin'

  const [searchQuery, setSearchQuery] = useState('')
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<DocumentListItem | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const documentsQuery = useDocumentsList()

  const filteredItems = useMemo(
    () => filterDocumentsBySearch(documentsQuery.items, searchQuery),
    [documentsQuery.items, searchQuery],
  )

  const handleRefresh = async () => {
    setIsRefreshing(true)
    try {
      await documentsQuery.refetch()
    } finally {
      setIsRefreshing(false)
    }
  }

  const handleView = (documentId: string) => {
    setSelectedDocumentId(documentId)
  }

  const handleDelete = (item: DocumentListItem) => {
    setDeleteTarget(item)
  }

  const handleDeleted = (documentId: string) => {
    setDeleteTarget(null)
    if (selectedDocumentId === documentId) {
      setSelectedDocumentId(null)
    }
    void documentsQuery.refetch()
  }

  if (user && !isAdmin) {
    return (
      <div className="space-y-8 max-w-7xl mx-auto pb-12">
        <TopBar
          title="Documents"
          description="Browse uploaded knowledge base files, track processing status, and manage document metadata."
        />
        <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-8 text-center space-y-3">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-destructive/20 text-destructive text-xl font-bold">
            !
          </div>
          <h2 className="text-xl font-bold text-foreground">Access Restricted</h2>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            You do not have permission to view or manage documents. Only administrators can access document management.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <TopBar
        title="Documents"
        description="Browse uploaded knowledge base files, track processing status, and manage document metadata."
      />

      <DocumentsToolbar
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onRefresh={() => {
          void handleRefresh()
        }}
        isRefreshing={isRefreshing || documentsQuery.isLoading}
        totalCount={documentsQuery.total}
        filteredCount={filteredItems.length}
      />

      {documentsQuery.isLoading ? <LoadingState rows={6} /> : null}

      {documentsQuery.isError ? (
        <ErrorState
          title="Could not load documents"
          error={documentsQuery.error}
          onRetry={() => {
            void handleRefresh()
          }}
        />
      ) : null}

      {!documentsQuery.isLoading && !documentsQuery.isError && filteredItems.length === 0 ? (
        <EmptyState
          title={searchQuery ? 'No matching documents' : 'No documents yet'}
          description={
            searchQuery
              ? 'Try a different title or filename search.'
              : 'Upload a document to populate your knowledge base.'
          }
          icon={<FileTextIcon className="size-5" />}
        />
      ) : null}

      {!documentsQuery.isLoading && !documentsQuery.isError && filteredItems.length > 0 ? (
        <>
          <DocumentsTable items={filteredItems} onView={handleView} onDelete={handleDelete} />
          <DocumentsCardList items={filteredItems} onView={handleView} onDelete={handleDelete} />
        </>
      ) : null}

      <DocumentDetailDrawer
        documentId={selectedDocumentId}
        open={Boolean(selectedDocumentId)}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedDocumentId(null)
          }
        }}
      />

      <DeleteDocumentDialog
        target={deleteTarget}
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!open) {
            setDeleteTarget(null)
          }
        }}
        onDeleted={handleDeleted}
      />
    </div>
  )
}
