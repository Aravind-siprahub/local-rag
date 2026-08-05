import type { DocumentDisplayStatus } from '@/types'
import { getDisplayStatusLabel } from '@/utils/documents'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

const statusStyles: Record<DocumentDisplayStatus, string> = {
  pending: 'border-primary/30 bg-primary/10 text-primary',
  parsing: 'border-warning/30 bg-warning/10 text-[hsl(var(--warning))]',
  chunked: 'border-secondary/30 bg-secondary/10 text-secondary',
  embedded: 'border-indigo-400/30 bg-indigo-400/10 text-indigo-300',
  ready: 'border-success/30 bg-success/10 text-[hsl(var(--success))]',
  failed: 'border-destructive/30 bg-destructive/10 text-destructive',
  archived: 'border-border bg-muted text-muted-foreground',
}

interface DocumentPipelineStatusBadgeProps {
  status: DocumentDisplayStatus
  className?: string
}

export function DocumentPipelineStatusBadge({
  status,
  className,
}: DocumentPipelineStatusBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn('rounded-full px-2.5 py-0.5 font-medium', statusStyles[status], className)}
    >
      {getDisplayStatusLabel(status)}
    </Badge>
  )
}
