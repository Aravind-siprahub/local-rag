import type { DocumentStatus } from '@/types'
import { getDocumentStatusLabel } from '@/utils/documents'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

const statusStyles: Record<DocumentStatus, string> = {
  uploaded: 'border-primary/30 bg-primary/10 text-primary',
  processing: 'border-warning/30 bg-warning/10 text-warning',
  ready: 'border-success/30 bg-success/10 text-[hsl(var(--success))]',
  failed: 'border-destructive/30 bg-destructive/10 text-destructive',
  archived: 'border-border bg-muted text-muted-foreground',
}

interface DocumentStatusBadgeProps {
  status: DocumentStatus
  className?: string
}

export function DocumentStatusBadge({ status, className }: DocumentStatusBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn('rounded-full px-2.5 py-0.5 font-medium', statusStyles[status], className)}
    >
      {getDocumentStatusLabel(status)}
    </Badge>
  )
}
