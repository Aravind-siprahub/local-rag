import { FileTextIcon } from 'lucide-react'
import { Link } from 'react-router-dom'

import { DocumentStatusBadge } from '@/components/DocumentStatusBadge'
import { EmptyState } from '@/components/EmptyState'
import { ErrorState } from '@/components/ErrorState'
import { LoadingState } from '@/components/LoadingState'
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
    <div className="flex flex-col rounded-xl border border-border/50 bg-card shadow-sm overflow-hidden">
      <div className="flex items-center justify-between border-b border-border/50 bg-muted/20 px-5 py-4">
        <h3 className="font-semibold tracking-tight">Recent documents</h3>
        {!isLoading && !isError && documents && documents.length > 0 && (
          <Link
            to={ROUTES.documents}
            className="text-sm font-medium text-primary hover:underline"
          >
            View all
          </Link>
        )}
      </div>

      <div className="p-0">
        {isLoading ? <div className="p-5"><LoadingState rows={4} /></div> : null}
        {isError ? <div className="p-5"><ErrorState title="Could not load documents" error={error} onRetry={onRetry} /></div> : null}
        
        {!isLoading && !isError && documents?.length === 0 ? (
          <div className="p-8">
            <EmptyState
              title="No documents yet"
              description="Upload your first document to start building your knowledge base."
              icon={<FileTextIcon className="size-6 text-muted-foreground" />}
            />
          </div>
        ) : null}

        {!isLoading && !isError && documents && documents.length > 0 ? (
          <div className="w-full">
            <div className="grid grid-cols-[1fr_auto_auto] items-center gap-4 border-b border-border/50 bg-muted/10 px-5 py-2 text-xs font-medium text-muted-foreground">
              <div>Document</div>
              <div className="w-24 text-center">Status</div>
              <div className="w-24 text-right">Added</div>
            </div>
            <ul className="divide-y divide-border/50">
              {documents.map((document) => (
                <li
                  key={document.id}
                  className="grid grid-cols-[1fr_auto_auto] items-center gap-4 px-5 py-3 transition-colors hover:bg-muted/10"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <FileTextIcon className="size-4 shrink-0 text-muted-foreground" />
                    <p className="truncate text-sm font-medium" title={document.title}>
                      {document.title}
                    </p>
                  </div>
                  <div className="w-24 text-center">
                    <DocumentStatusBadge status={document.status} />
                  </div>
                  <div className="w-24 text-right text-xs text-muted-foreground">
                    {formatRelativeTime(document.created_at)}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  )
}
