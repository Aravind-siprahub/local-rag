import { FileTextIcon } from 'lucide-react'
import { useState } from 'react'

import {
  UnsupportedFileDialog,
  UploadDropzone,
  UploadHeader,
  UploadQueue,
  useUploadQueue,
} from '@/features/upload'
import { useAuth } from '@/features/auth/hooks/authHooks'

export function UploadPage() {
  const { user } = useAuth()
  const isAdmin = user?.role?.toLowerCase() === 'admin'

  const {
    queue,
    rejectedFiles,
    primaryUser,
    isUserLoading,
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
  } = useUploadQueue()

  const [isDialogOpen, setIsDialogOpen] = useState(true)

  // Backend is reachable AND we have a user to associate uploads with
  const hasBackendAvailable = isBackendReachable && Boolean(primaryUser) && !isUserLoading

  if (user && !isAdmin) {
    return (
      <div className="space-y-8 max-w-7xl mx-auto pb-12">
        <UploadHeader />
        <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-8 text-center space-y-3">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-destructive/20 text-destructive text-xl font-bold">
            !
          </div>
          <h2 className="text-xl font-bold text-foreground">Access Restricted</h2>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            You do not have permission to upload documents. Only administrators can upload documents to the knowledge base.
          </p>
        </div>
      </div>
    )
  }

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
            isBackendReachable={isBackendReachable}
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
