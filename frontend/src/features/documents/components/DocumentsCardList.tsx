import { EyeIcon, Trash2Icon } from 'lucide-react'

import { DocumentPipelineStatusBadge } from '@/features/documents/components/DocumentPipelineStatusBadge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { DocumentListItem } from '@/types'
import { formatDateTime, formatFileSize } from '@/utils'

interface DocumentsCardListProps {
  items: DocumentListItem[]
  onView: (documentId: string) => void
  onDelete: (item: DocumentListItem) => void
}

export function DocumentsCardList({ items, onView, onDelete }: DocumentsCardListProps) {
  return (
    <div className="grid gap-4 md:hidden">
      {items.map((item) => (
        <Card
          key={item.document.id}
          className="glass-panel glass-panel-hover cursor-pointer border-border/60"
          onClick={() => onView(item.document.id)}
        >
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <CardTitle className="truncate text-base">{item.document.title}</CardTitle>
                <p className="mt-1 truncate text-sm text-muted-foreground">
                  {item.filename ?? 'No file attached'}
                </p>
              </div>
              <DocumentPipelineStatusBadge status={item.displayStatus} />
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-xs text-muted-foreground">Version</p>
                <p>{item.versionLabel ?? '—'}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Size</p>
                <p>{formatFileSize(item.fileSizeBytes)}</p>
              </div>
              <div className="col-span-2">
                <p className="text-xs text-muted-foreground">Created</p>
                <p>{formatDateTime(item.document.created_at)}</p>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="flex-1"
                onClick={(event) => {
                  event.stopPropagation()
                  onView(item.document.id)
                }}
              >
                <EyeIcon />
                View
              </Button>
              <Button
                type="button"
                variant="destructive"
                size="sm"
                onClick={(event) => {
                  event.stopPropagation()
                  onDelete(item)
                }}
              >
                <Trash2Icon />
              </Button>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
