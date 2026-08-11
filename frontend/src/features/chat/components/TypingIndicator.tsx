interface TypingIndicatorProps {
  label?: string
}

export function TypingIndicator({ label = 'Generating response...' }: TypingIndicatorProps) {
  return (
    <div className="flex flex-col gap-2 p-4 bg-muted/50 rounded-lg max-w-xs mb-4">
      <div className="flex items-center space-x-1">
        <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  )
}
