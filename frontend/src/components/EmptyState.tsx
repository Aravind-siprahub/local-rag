import { InboxIcon } from 'lucide-react'

interface EmptyStateProps {
  title: string
  description: string
  icon?: React.ReactNode
}

export function EmptyState({ title, description, icon }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border/80 bg-muted/20 px-6 py-10 text-center">
      <div className="flex size-12 items-center justify-center rounded-full bg-muted/60 text-muted-foreground">
        {icon ?? <InboxIcon className="size-5" />}
      </div>
      <div className="space-y-1">
        <p className="font-medium text-foreground">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
    </div>
  )
}
