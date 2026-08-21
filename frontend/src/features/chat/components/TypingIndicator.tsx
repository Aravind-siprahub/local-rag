interface TypingIndicatorProps {
  label?: string
}

export function TypingIndicator({ label = 'Generating response...' }: TypingIndicatorProps) {
  return (
    <div className="flex items-center gap-3 p-1.5 text-muted-foreground/60 select-none animate-in fade-in-0 duration-200">
      <div className="flex items-center gap-1 bg-muted/40 border border-border/20 px-2.5 py-1.5 rounded-xl shadow-xs">
        <div className="w-1.5 h-1.5 bg-primary/70 rounded-full animate-bounce" style={{ animationDelay: '0ms', animationDuration: '1s' }} />
        <div className="w-1.5 h-1.5 bg-primary/70 rounded-full animate-bounce" style={{ animationDelay: '150ms', animationDuration: '1s' }} />
        <div className="w-1.5 h-1.5 bg-primary/70 rounded-full animate-bounce" style={{ animationDelay: '300ms', animationDuration: '1s' }} />
      </div>
      <span className="text-[11px] font-medium animate-pulse">{label}</span>
    </div>
  )
}
