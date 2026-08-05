import { MessageSquare } from 'lucide-react'

interface EmptyStateProps {
  title?: string
  description?: string
}

export function ChatEmptyState({
  title = 'No messages yet',
  description = 'Start a conversation to ask questions grounded in your documents.',
}: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8 text-muted-foreground">
      <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
        <MessageSquare className="w-6 h-6 text-primary" />
      </div>
      <h3 className="text-lg font-medium text-foreground mb-1">{title}</h3>
      <p className="text-sm max-w-sm">{description}</p>
    </div>
  )
}
