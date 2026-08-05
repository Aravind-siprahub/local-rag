export function TypingIndicator() {
  return (
    <div className="flex items-center space-x-1 p-4 bg-muted/50 rounded-lg max-w-25 mb-4">
      <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
      <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
      <div className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
  )
}
