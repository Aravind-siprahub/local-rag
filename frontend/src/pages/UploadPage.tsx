import { FileTextIcon, RefreshCwIcon } from 'lucide-react'

import { EmptyState } from '@/components/EmptyState'
import { ErrorState } from '@/components/ErrorState'
import { LoadingState } from '@/components/LoadingState'
import { Button } from '@/components/ui/button'
import { DocumentsCardList, DocumentsTable } from '@/features/documents/components'
import { useDocumentsList } from '@/features/documents/hooks'
import {
  UnsupportedFileDialog,
  UploadDropzone,
  UploadHeader,
  UploadQueue,
  useUploadQueue,
} from '@/features/upload'
import { useState } from 'react'

export function UploadPage() {
  const {
    queue,
    rejectedFiles,
    primaryUser,
    isUserLoading,
    isUploading,
    overallProgress,
    addFiles,
    removeItem,
    retryItem,
    cancelItem,
    clearCompleted,
    clearRejected,
    uploadAll,
  } = useUploadQueue()

  const [isDialogOpen, setIsDialogOpen] = useState(true)

  // Recent Uploads query reusing documents feature list
  const documentsQuery = useDocumentsList()

  const hasBackendAvailable = Boolean(primaryUser) && !isUserLoading

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      <UploadHeader />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Dropzone & Queue */}
        <div className="lg:col-span-7 space-y-6">
          <UploadDropzone onFilesSelected={addFiles} disabled={!hasBackendAvailable} />

          <UploadQueue
            queue={queue}
            isUploading={isUploading}
            overallProgress={overallProgress}
            hasBackendAvailable={hasBackendAvailable}
            onUploadAll={() => {
              void uploadAll()
            }}
            onRetry={retryItem}
            onCancel={cancelItem}
            onRemove={removeItem}
            onClearCompleted={clearCompleted}
          />
        </div>

        {/* Right Column: Ingestion Guidelines & Status */}
        <div className="lg:col-span-5 space-y-6">
          <div className="p-6 rounded-xl border border-border/60 bg-card/60 space-y-4">
            <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
              <FileTextIcon className="size-4 text-primary" /> Ingestion Guidelines
            </h3>
            <ul className="text-xs text-muted-foreground space-y-2.5 list-disc list-inside leading-relaxed">
              <li>
                <strong className="text-foreground">Text Extraction:</strong> Documents are automatically parsed into raw text during the worker job pass.
              </li>
              <li>
                <strong className="text-foreground">Chunking Strategy:</strong> Text is split into <code className="font-mono text-foreground">1,000</code> character chunks with <code className="font-mono text-foreground">200</code> character overlap.
              </li>
              <li>
                <strong className="text-foreground">Vector Embeddings:</strong> Embeddings are generated locally using <code className="font-mono text-foreground">nomic-embed-text</code> (768 dimensions).
              </li>
              <li>
                <strong className="text-foreground">File Size Limits:</strong> Individual uploads are capped at 25 MB per document file.
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Recent Uploads Section */}
      <div className="space-y-4 pt-4 border-t border-border/40">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Recent Knowledge Base Ingestions</h2>
            <p className="text-xs text-muted-foreground">
              Review recently processed documents and their pipeline statuses in your local database.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              void documentsQuery.refetch()
            }}
            disabled={documentsQuery.isLoading}
          >
            <RefreshCwIcon className={`size-3.5 mr-1.5 ${documentsQuery.isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {documentsQuery.isLoading ? <LoadingState rows={4} /> : null}

        {documentsQuery.isError ? (
          <ErrorState
            title="Could not load recent uploads"
            error={documentsQuery.error}
            onRetry={() => {
              void documentsQuery.refetch()
            }}
          />
        ) : null}

        {!documentsQuery.isLoading && !documentsQuery.isError && documentsQuery.items.length === 0 ? (
          <EmptyState
            title="No uploaded documents yet"
            description="Use the dropzone above to queue and ingest your first document file."
            icon={<FileTextIcon className="size-5" />}
          />
        ) : null}

        {!documentsQuery.isLoading && !documentsQuery.isError && documentsQuery.items.length > 0 ? (
          <>
            <DocumentsTable
              items={documentsQuery.items.slice(0, 5)}
              onView={() => {}}
              onDelete={() => {}}
            />
            <DocumentsCardList
              items={documentsQuery.items.slice(0, 5)}
              onView={() => {}}
              onDelete={() => {}}
            />
          </>
        ) : null}
      </div>

      {/* Unsupported File Warning Dialog */}
      <UnsupportedFileDialog
        rejectedFiles={rejectedFiles}
        open={rejectedFiles.length > 0 && isDialogOpen}
        onOpenChange={setIsDialogOpen}
        onClear={clearRejected}
      />
    </div>
  )
}
