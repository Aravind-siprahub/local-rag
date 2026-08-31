import type { HealthResponse } from '@/types'
import { Skeleton } from '@/components/ui/skeleton'

interface KnowledgeHealthProps {
  health?: HealthResponse
  isLoading: boolean
}

export function KnowledgeHealth({ health, isLoading }: KnowledgeHealthProps) {
  return (
    <div className="flex flex-col rounded-xl border border-border/50 bg-card shadow-sm overflow-hidden h-full">
      <div className="border-b border-border/50 bg-muted/20 px-5 py-4">
        <h3 className="font-semibold tracking-tight">Knowledge base health</h3>
      </div>
      <div className="p-5">
        {isLoading ? (
          <div className="space-y-4">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-full" />
          </div>
        ) : (
          <ul className="space-y-4">
            <HealthItem
              label="API"
              status={health?.status === 'ok' ? 'ok' : 'error'}
              text={health?.status === 'ok' ? 'Online' : 'Offline'}
            />
            <HealthItem
              label="Database"
              status={health?.database === 'connected' ? 'ok' : 'error'}
              text={health?.database === 'connected' ? 'Connected' : 'Disconnected'}
            />
            <HealthItem
              label="Vector search"
              status={health?.pgvector === 'installed' ? 'ok' : 'error'}
              text={health?.pgvector === 'installed' ? 'Ready' : 'Not installed'}
            />
            <HealthItem
              label="Ollama"
              status={health?.ollama === 'available' ? 'ok' : 'error'}
              text={health?.ollama === 'available' ? 'Online' : 'Offline'}
            />
            <HealthItem
              label="Models"
              status={(health?.models && health.models.length > 0) ? 'ok' : 'error'}
              text={(health?.models && health.models.length > 0) ? 'Ready' : 'Not found'}
            />
          </ul>
        )}
      </div>
    </div>
  )
}

function HealthItem({
  label,
  status,
  text,
}: {
  label: string
  status: 'ok' | 'error'
  text: string
}) {
  const isOk = status === 'ok'
  return (
    <li className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className={`size-2 rounded-full ${isOk ? 'bg-success' : 'bg-destructive'}`} />
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
      <span className={`text-sm font-medium ${isOk ? 'text-foreground' : 'text-destructive'}`}>
        {text}
      </span>
    </li>
  )
}
