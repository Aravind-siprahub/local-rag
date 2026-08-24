import {
  AlertTriangleIcon,
  CheckCircle2Icon,
  FileStackIcon,
  LoaderCircleIcon,
} from 'lucide-react'

import { Skeleton } from '@/components/ui/skeleton'
import type { DocumentStats } from '@/types'

interface DashboardStatsProps {
  stats?: DocumentStats
  isLoading: boolean
}

export function DashboardStats({ stats, isLoading }: DashboardStatsProps) {
  if (isLoading) {
    return <Skeleton className="h-24 w-full rounded-xl" />
  }

  const total = stats?.total ?? 0
  const ready = stats?.ready ?? 0
  const processing = stats?.processing ?? 0
  const failed = stats?.failed ?? 0
  const archived = stats?.archived ?? 0

  const readyPercentage = total > 0 ? Math.round((ready / total) * 100) : 0

  return (
    <div className="flex flex-col sm:flex-row divide-y sm:divide-y-0 sm:divide-x divide-border/50 rounded-xl border border-border/50 bg-card shadow-sm">
      <div className="flex-1 p-5 flex flex-col gap-1">
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <FileStackIcon className="size-4 text-primary" />
          Documents
        </div>
        <div className="text-3xl font-semibold tracking-tight">{total}</div>
        <div className="text-xs text-muted-foreground">
          {archived > 0 ? `${archived} archived` : 'Total in knowledge base'}
        </div>
      </div>

      <div className="flex-1 p-5 flex flex-col gap-1">
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <CheckCircle2Icon className="size-4 text-success" />
          Ready
        </div>
        <div className="text-3xl font-semibold tracking-tight">{ready}</div>
        <div className="text-xs text-muted-foreground">
          {total > 0 ? `${readyPercentage}% ready for RAG` : 'No documents yet'}
        </div>
      </div>

      <div className="flex-1 p-5 flex flex-col gap-1">
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <LoaderCircleIcon className="size-4 text-warning" />
          Processing
        </div>
        <div className="text-3xl font-semibold tracking-tight">{processing}</div>
        <div className="text-xs text-muted-foreground">
          {processing > 0 ? 'Currently ingesting...' : 'All clear'}
        </div>
      </div>

      <div className="flex-1 p-5 flex flex-col gap-1">
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <AlertTriangleIcon className="size-4 text-destructive" />
          Failed
        </div>
        <div className="text-3xl font-semibold tracking-tight">{failed}</div>
        <div className="text-xs text-muted-foreground">
          {failed > 0 ? 'Requires attention' : 'No errors'}
        </div>
      </div>
    </div>
  )
}
