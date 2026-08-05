import { AlertCircleIcon, Loader2Icon } from 'lucide-react'

import { Badge } from '@/components/ui/badge'

interface HealthBadgeProps {
  status: string | null
  isLoading?: boolean
  isError?: boolean
}

export function HealthBadge({ status, isLoading = false, isError = false }: HealthBadgeProps) {
  if (isLoading) {
    return (
      <Badge variant="outline" className="bg-muted/40 text-muted-foreground border-border/60">
        <Loader2Icon className="size-3 animate-spin mr-1 inline" /> Checking...
      </Badge>
    )
  }

  if (isError || status !== 'ok') {
    return (
      <Badge variant="destructive" className="bg-destructive/10 text-destructive border-destructive/20">
        <AlertCircleIcon className="size-3 mr-1 inline" /> Offline / Error
      </Badge>
    )
  }

  return (
    <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20 font-medium">
      <span className="size-2 rounded-full bg-emerald-500 animate-pulse mr-1.5 inline-block" />
      Healthy (Connected)
    </Badge>
  )
}
