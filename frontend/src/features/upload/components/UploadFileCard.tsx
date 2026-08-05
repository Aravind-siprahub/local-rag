import {
  FileTextIcon,
  FileIcon,
  CheckCircle2Icon,
  AlertCircleIcon,
  RotateCcwIcon,
  XIcon,
  Loader2Icon,
} from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { UploadQueueItem, UploadQueueStatus } from '@/types'
import { formatFileSize } from '@/utils'

interface UploadFileCardProps {
  item: UploadQueueItem
  onRetry: (id: string) => void
  onCancel: (id: string) => void
  onRemove: (id: string) => void
}

export function UploadFileCard({ item, onRetry, onCancel, onRemove }: UploadFileCardProps) {
  const isUploading = item.status === 'Uploading'
  const isFailed = item.status === 'Failed'
  const isReady = item.status === 'Ready'

  const renderStatusBadge = (status: UploadQueueStatus) => {
    switch (status) {
      case 'Waiting':
        return (
          <Badge variant="outline" className="bg-muted/40 text-muted-foreground border-border/60">
            Waiting
          </Badge>
        )
      case 'Uploading':
        return (
          <Badge variant="secondary" className="bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20 animate-pulse">
            <Loader2Icon className="size-3 animate-spin mr-1 inline" /> Uploading {item.progress}%
          </Badge>
        )
      case 'Parsing':
        return (
          <Badge variant="secondary" className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20">
            <Loader2Icon className="size-3 animate-spin mr-1 inline" /> Parsing
          </Badge>
        )
      case 'Chunking':
        return (
          <Badge variant="secondary" className="bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20">
            <Loader2Icon className="size-3 animate-spin mr-1 inline" /> Chunking
          </Badge>
        )
      case 'Embedding':
        return (
          <Badge variant="secondary" className="bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/20">
            <Loader2Icon className="size-3 animate-spin mr-1 inline" /> Embedding
          </Badge>
        )
      case 'Ready':
        return (
          <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20">
            <CheckCircle2Icon className="size-3 mr-1 inline" /> Ready
          </Badge>
        )
      case 'Failed':
        return (
          <Badge variant="destructive" className="bg-destructive/10 text-destructive border-destructive/20">
            <AlertCircleIcon className="size-3 mr-1 inline" /> Failed
          </Badge>
        )
    }
  }

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase()
    if (ext === 'pdf') {
      return <FileTextIcon className="size-5 text-red-500" />
    }
    if (ext === 'docx') {
      return <FileTextIcon className="size-5 text-blue-500" />
    }
    if (ext === 'md' || ext === 'markdown') {
      return <FileTextIcon className="size-5 text-teal-500" />
    }
    return <FileIcon className="size-5 text-muted-foreground" />
  }

  return (
    <div className="p-4 rounded-lg border border-border/60 bg-card hover:bg-card/80 transition-colors space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <div className="p-2 rounded-md bg-muted/50 shrink-0">{getFileIcon(item.name)}</div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium text-foreground truncate max-w-70 sm:max-w-xs md:max-w-md">
                {item.name}
              </span>
              <span className="text-xs text-muted-foreground font-mono">
                ({formatFileSize(item.size)})
              </span>
            </div>

            {item.error ? (
              <p className="text-xs text-destructive mt-0.5 font-medium leading-tight">
                {item.error}
              </p>
            ) : null}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {renderStatusBadge(item.status)}

          {isFailed ? (
            <Button
              type="button"
              variant="outline"
              size="icon-xs"
              onClick={() => onRetry(item.id)}
              title="Retry upload"
            >
              <RotateCcwIcon className="size-3.5" />
            </Button>
          ) : null}

          {isUploading ? (
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              onClick={() => onCancel(item.id)}
              title="Cancel upload"
            >
              <XIcon className="size-3.5 text-muted-foreground" />
            </Button>
          ) : (
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              onClick={() => onRemove(item.id)}
              title="Remove from queue"
            >
              <XIcon className="size-3.5 text-muted-foreground hover:text-destructive" />
            </Button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {isUploading || item.status === 'Parsing' || item.status === 'Chunking' || item.status === 'Embedding' ? (
        <div className="space-y-1">
          <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 rounded-full ${
                isReady ? 'bg-emerald-500' : 'bg-primary'
              }`}
              style={{ width: `${item.progress}%` }}
            />
          </div>
        </div>
      ) : null}
    </div>
  )
}
