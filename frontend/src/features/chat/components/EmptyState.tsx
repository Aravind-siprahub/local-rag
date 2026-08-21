import { MessageSquare } from 'lucide-react'

interface EmptyStateProps {
  title?: string
  description?: string
}

export function ChatEmptyState({
  title = 'Knowledge Studio Chat',
  description = 'Ask questions grounded in your uploaded documents and analyze them with local AI models.',
}: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-6 select-none animate-in fade-in-0 duration-300">
      <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-4.5 shadow-xs">
        <MessageSquare className="w-5 h-5 text-primary" />
      </div>
      <h3 className="text-base font-bold text-foreground font-display mb-1.5">{title}</h3>
      <p className="text-xs text-muted-foreground/80 max-w-sm leading-relaxed">{description}</p>
    </div>
  )
}
