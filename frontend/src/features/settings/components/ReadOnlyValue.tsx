import { LockIcon } from 'lucide-react'

import { Badge } from '@/components/ui/badge'

interface ReadOnlyValueProps {
  value: string | number | boolean
  reason?: string
}

export function ReadOnlyValue({ value, reason = 'Configured in backend .env' }: ReadOnlyValueProps) {
  const displayVal = typeof value === 'boolean' ? (value ? 'True' : 'False') : String(value)

  return (
    <div className="flex items-center gap-2">
      <Badge variant="outline" className="font-mono text-xs bg-muted/30 px-2.5 py-1">
        {displayVal}
      </Badge>
      <span className="text-[11px] text-muted-foreground/70 flex items-center gap-1" title={reason}>
        <LockIcon className="size-3" />
        Read-only
      </span>
    </div>
  )
}
