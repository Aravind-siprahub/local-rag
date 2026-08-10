import { Trash2Icon, UploadIcon, Loader2Icon } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { EmptyUploadState } from '@/features/upload/components/EmptyUploadState'
import { UploadFileCard } from '@/features/upload/components/UploadFileCard'
import { UploadProgress } from '@/features/upload/components/UploadProgress'
import type { UploadQueueItem } from '@/types'

interface UploadQueueProps {
  queue: UploadQueueItem[]
  isUploading: boolean
  overallProgress: number
  hasBackendAvailable: boolean
  isBackendReachable: boolean
  onUploadAll: () => void
  onRetry: (id: string) => void
  onCancel: (id: string) => void
  onRemove: (id: string) => void
  onClearCompleted: () => void
}

export function UploadQueue({
  queue,
  isUploading,
  overallProgress,
  hasBackendAvailable,
  isBackendReachable,
  onUploadAll,
  onRetry,
  onCancel,
  onRemove,
  onClearCompleted,
}: UploadQueueProps) {
  const waitingOrFailedCount = queue.filter(
    (item) => item.status === 'Waiting' || item.status === 'Failed',
  ).length
  const completedCount = queue.filter((item) => item.status === 'Ready').length
  const failedCount = queue.filter((item) => item.status === 'Failed').length

  const canStartUpload = hasBackendAvailable && waitingOrFailedCount > 0 && !isUploading

  return (
    <div className="space-y-4">
      {!isBackendReachable ? (
        <Alert variant="destructive" className="bg-destructive/10 text-destructive border-destructive/20">
          <AlertTitle className="font-semibold text-sm">Upload backend is not yet available.</AlertTitle>
          <AlertDescription className="text-xs mt-1">
            The document ingestion API endpoint is unreachable. Make sure the backend server is running.
          </AlertDescription>
        </Alert>
      ) : !hasBackendAvailable ? (
        <Alert className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20">
          <AlertTitle className="font-semibold text-sm">No active user found.</AlertTitle>
          <AlertDescription className="text-xs mt-1">
            Create a user first via <strong>Settings → User Management</strong> (or <code>POST /api/users</code>), then return here to upload documents.
          </AlertDescription>
        </Alert>
      ) : null}

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-border/40">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Upload Queue</h2>
          <p className="text-xs text-muted-foreground">
            {queue.length === 0
              ? 'No files currently queued.'
              : `${queue.length} file${queue.length === 1 ? '' : 's'} in queue`}
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {completedCount > 0 ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onClearCompleted}
              disabled={isUploading}
            >
              <Trash2Icon className="size-3.5 mr-1.5" />
              Clear Completed
            </Button>
          ) : null}

          <Button
            type="button"
            variant="default"
            size="sm"
            onClick={onUploadAll}
            disabled={!canStartUpload}
            className="shadow-2xs"
          >
            {isUploading ? (
              <>
                <Loader2Icon className="size-3.5 mr-1.5 animate-spin" />
                Uploading Queue...
              </>
            ) : (
              <>
                <UploadIcon className="size-3.5 mr-1.5" />
                Upload All ({waitingOrFailedCount})
              </>
            )}
          </Button>
        </div>
      </div>

      {queue.length > 0 ? (
        <div className="space-y-4">
          <UploadProgress
            overallProgress={overallProgress}
            totalFiles={queue.length}
            completedCount={completedCount}
            failedCount={failedCount}
            isUploading={isUploading}
          />

          <div className="space-y-2 max-h-105 overflow-y-auto pr-1">
            {queue.map((item) => (
              <UploadFileCard
                key={item.id}
                item={item}
                onRetry={onRetry}
                onCancel={onCancel}
                onRemove={onRemove}
              />
            ))}
          </div>
        </div>
      ) : (
        <EmptyUploadState />
      )}
    </div>
  )
}
