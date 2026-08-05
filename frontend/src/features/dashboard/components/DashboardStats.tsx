import {
  AlertTriangleIcon,
  ArchiveIcon,
  CheckCircle2Icon,
  FileStackIcon,
  LoaderCircleIcon,
} from 'lucide-react'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import type { DocumentStats } from '@/types'

interface DashboardStatsProps {
  stats?: DocumentStats
  isLoading: boolean
}

const statCards = [
  {
    key: 'total' as const,
    label: 'Total documents',
    icon: FileStackIcon,
    accent: 'from-primary/20 to-primary/5 text-primary',
  },
  {
    key: 'ready' as const,
    label: 'Ready for RAG',
    icon: CheckCircle2Icon,
    accent: 'from-[hsl(var(--success)/0.2)] to-[hsl(var(--success)/0.05)] text-[hsl(var(--success))]',
  },
  {
    key: 'processing' as const,
    label: 'Processing',
    icon: LoaderCircleIcon,
    accent: 'from-[hsl(var(--warning)/0.2)] to-[hsl(var(--warning)/0.05)] text-[hsl(var(--warning))]',
  },
  {
    key: 'failed' as const,
    label: 'Failed',
    icon: AlertTriangleIcon,
    accent: 'from-destructive/20 to-destructive/5 text-destructive',
  },
  {
    key: 'archived' as const,
    label: 'Archived',
    icon: ArchiveIcon,
    accent: 'from-muted/80 to-muted/30 text-muted-foreground',
  },
]

export function DashboardStats({ stats, isLoading }: DashboardStatsProps) {
  if (isLoading) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {statCards.map((card) => (
          <Skeleton key={card.key} className="h-32 rounded-xl" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
      {statCards.map(({ key, label, icon: Icon, accent }) => (
        <Card key={key} className="glass-panel glass-panel-hover border-border/60">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
              <div
                className={`flex size-9 items-center justify-center rounded-lg bg-gradient-to-br ${accent}`}
              >
                <Icon className="size-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-semibold tracking-tight">{stats?.[key] ?? 0}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
