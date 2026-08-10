import { LoaderCircleIcon } from 'lucide-react'

import { ErrorState } from '@/components/ErrorState'
import { DocumentPipelineStatusBadge } from '@/features/documents/components/DocumentPipelineStatusBadge'
import { useDocumentDetail } from '@/features/documents/hooks'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { formatDateTime, formatFileSize, getPipelineProgress, getDisplayStatusLabel } from '@/utils'

interface DocumentDetailDrawerProps {
  documentId: string | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

function DetailField({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <div className="text-sm text-foreground">{value}</div>
    </div>
  )
}

export function DocumentDetailDrawer({
  documentId,
  open,
  onOpenChange,
}: DocumentDetailDrawerProps) {
  const detail = useDocumentDetail(documentId)

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full border-border/60 sm:max-w-lg">
        <SheetHeader className="border-b border-border/60 pb-4">
          <SheetTitle>{detail.document?.title ?? 'Document details'}</SheetTitle>
          <SheetDescription>
            Metadata, versions, and processing status from the backend.
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="h-[calc(100vh-8rem)] pr-4">
          {detail.isLoading ? (
            <div className="space-y-4 py-4">
              <Skeleton className="h-24 w-full rounded-xl" />
              <Skeleton className="h-32 w-full rounded-xl" />
              <Skeleton className="h-40 w-full rounded-xl" />
            </div>
          ) : null}

          {detail.isError ? (
            <div className="py-4">
              <ErrorState
                title="Could not load document details"
                error={detail.error}
                onRetry={() => {
                  void detail.refetch()
                }}
              />
            </div>
          ) : null}

          {!detail.isLoading && !detail.isError && detail.document ? (
            <div className="space-y-6 py-4">
              <section className="glass-panel space-y-4 rounded-xl border border-border/60 p-4">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="font-medium">Metadata</h3>
                  {detail.listItem ? (
                    <DocumentPipelineStatusBadge status={detail.listItem.displayStatus} />
                  ) : null}
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <DetailField label="Title" value={detail.document.title} />
                  <DetailField
                    label="Filename"
                    value={detail.listItem?.filename ?? '—'}
                  />
                  <DetailField
                    label="Pipeline status"
                    value={
                      detail.listItem
                        ? getDisplayStatusLabel(detail.listItem.displayStatus)
                        : '—'
                    }
                  />
                  <DetailField
                    label="File size"
                    value={formatFileSize(detail.listItem?.fileSizeBytes)}
                  />
                  <DetailField
                    label="Created"
                    value={formatDateTime(detail.document.created_at)}
                  />
                  <DetailField
                    label="Updated"
                    value={formatDateTime(detail.document.updated_at)}
                  />
                  <DetailField
                    label="Tags"
                    value={
                      detail.document.tags.length > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {detail.document.tags.map((tag) => (
                            <Badge key={tag} variant="outline">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        '—'
                      )
                    }
                  />
                  <DetailField
                    label="Description"
                    value={detail.document.description ?? '—'}
                  />
                </div>
              </section>

              <section className="glass-panel space-y-4 rounded-xl border border-border/60 p-4">
                <h3 className="font-medium">Processing status</h3>
                {detail.currentVersion ? (
                  <div className="grid gap-4 sm:grid-cols-2">
                    <DetailField
                      label="Version status"
                      value={detail.currentVersion.status.replaceAll('_', ' ')}
                    />
                    <DetailField
                      label="Parsed"
                      value={formatDateTime(detail.currentVersion.parsed_at)}
                    />
                    <DetailField
                      label="Chunked"
                      value={formatDateTime(detail.currentVersion.chunked_at)}
                    />
                    <DetailField
                      label="Embedded"
                      value={formatDateTime(detail.currentVersion.embedded_at)}
                    />
                    <DetailField
                      label="Chunks"
                      value={
                        getPipelineProgress(detail.currentVersion).chunksCreated
                          ? 'Created'
                          : 'Pending'
                      }
                    />
                    <DetailField
                      label="Embeddings"
                      value={
                        getPipelineProgress(detail.currentVersion).embeddingsCreated
                          ? 'Created'
                          : 'Pending'
                      }
                    />
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No version data available.</p>
                )}

                {detail.processingJobs.length > 0 ? (
                  <>
                    <Separator />
                    <div className="space-y-2">
                      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Processing jobs
                      </p>
                      <ul className="space-y-2">
                        {Object.values(
                          detail.processingJobs.reduce<Record<string, (typeof detail.processingJobs)[number]>>(
                            (acc, job) => {
                              // Keep the latest/most active job for each job type
                              if (!acc[job.job_type] || job.status === 'running' || new Date(job.created_at) > new Date(acc[job.job_type].created_at)) {
                                acc[job.job_type] = job
                              }
                              return acc
                            },
                            {},
                          ),
                        ).map((job) => (
                          <li
                            key={job.id}
                            className="flex items-center justify-between rounded-lg border border-border/60 bg-background/40 px-3 py-2 text-sm"
                          >
                            <span className="capitalize flex items-center">
                              {job.job_type}
                              {job.status === 'running' ? (
                                <LoaderCircleIcon className="ml-2 size-3.5 animate-spin text-primary" />
                              ) : null}
                            </span>
                            <Badge variant="outline" className="capitalize">
                              {job.status}
                            </Badge>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </>
                ) : null}

                {detail.currentVersion?.error_message ? (
                  <p className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                    {detail.currentVersion.error_message}
                  </p>
                ) : null}
              </section>

              <section className="glass-panel space-y-4 rounded-xl border border-border/60 p-4">
                <h3 className="font-medium">Versions ({detail.versions.length})</h3>
                {detail.versions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No versions uploaded yet.</p>
                ) : (
                  <ul className="space-y-3">
                    {[...detail.versions]
                      .sort((left, right) => right.version_number - left.version_number)
                      .map((version) => (
                        <li
                          key={version.id}
                          className="rounded-lg border border-border/60 bg-background/40 px-3 py-3 text-sm"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div>
                              <p className="font-medium">Version {version.version_number}</p>
                              <p className="text-muted-foreground">{version.original_filename}</p>
                            </div>
                            <Badge variant="outline" className="capitalize">
                              {version.status}
                            </Badge>
                          </div>
                          <p className="mt-2 text-xs text-muted-foreground">
                            {formatFileSize(version.file_size_bytes)} ·{' '}
                            {formatDateTime(version.created_at)}
                          </p>
                        </li>
                      ))}
                  </ul>
                )}
              </section>
            </div>
          ) : null}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}
