import { FileTextIcon } from 'lucide-react'
import { Link } from 'react-router-dom'

import { DocumentStatusBadge } from '@/components/DocumentStatusBadge'
import { EmptyState } from '@/components/EmptyState'
import { ErrorState } from '@/components/ErrorState'
import { LoadingState } from '@/components/LoadingState'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { Document } from '@/types'
import { formatRelativeTime } from '@/utils/date'
import { ROUTES } from '@/routes/paths'

interface RecentUploadsProps {
  documents?: Document[]
  isLoading: boolean
  isError: boolean
  error: unknown
  onRetry: () => void
}

export function RecentUploads({
  documents,
  isLoading,
  isError,
  error,
  onRetry,
}: RecentUploadsProps) {
  return (
    <Card className="glass-panel border-border/60">
      <CardHeader>
        <CardTitle>Recent uploads</CardTitle>
        <CardDescription>Latest documents indexed in your knowledge base.</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? <LoadingState rows={5} /> : null}
        {isError ? <ErrorState title="Could not load documents" error={error} onRetry={onRetry} /> : null}
        {!isLoading && !isError && documents?.length === 0 ? (
          <EmptyState
            title="No documents yet"
            description="Upload your first document to start building your knowledge base."
            icon={<FileTextIcon className="size-5" />}
          />
        ) : null}
        {!isLoading && !isError && documents && documents.length > 0 ? (
          <ul className="space-y-3">
            {documents.map((document) => (
              <li
                key={document.id}
                className="flex items-start justify-between gap-4 rounded-xl border border-border/60 bg-background/40 px-4 py-3 transition-colors hover:border-primary/20"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-foreground">{document.title}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Uploaded {formatRelativeTime(document.created_at)}
                  </p>
                </div>
                <DocumentStatusBadge status={document.status} />
              </li>
            ))}
          </ul>
        ) : null}

        {!isLoading && !isError && documents && documents.length > 0 ? (
          <Link
            to={ROUTES.documents}
            className="mt-4 inline-flex text-sm font-medium text-primary hover:underline"
          >
            View all documents
          </Link>
        ) : null}
      </CardContent>
    </Card>
  )
}
