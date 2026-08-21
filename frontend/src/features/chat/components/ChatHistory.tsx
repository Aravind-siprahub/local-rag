import { useEffect, useRef } from 'react'
import { ChatMessage } from './ChatMessage'
import { TypingIndicator } from './TypingIndicator'
import { ChatEmptyState } from './EmptyState'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { Message, Citation } from '../types/chat'

interface ChatHistoryProps {
  messages: Message[]
  isLoading?: boolean
  latestCitations?: Citation[]
  loadingLabel?: string
  onEditMessage?: (message: Message) => void
  onRegenerateMessage?: (message: Message) => void
  onSuggestedClick?: (text: string) => void
}

export function ChatHistory({
  messages,
  isLoading,
  latestCitations,
  loadingLabel,
  onEditMessage,
  onRegenerateMessage,
  onSuggestedClick,
}: ChatHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Use requestAnimationFrame/setTimeout to ensure DOM has updated with the latest streamed text
    const timer = setTimeout(() => {
      if (bottomRef.current) {
        bottomRef.current.scrollIntoView({ behavior: 'auto', block: 'end' })
      } else if (scrollRef.current) {
        const viewport =
          scrollRef.current.querySelector('[data-radix-scroll-area-viewport]') || scrollRef.current
        viewport.scrollTop = viewport.scrollHeight
      }
    }, 50)
    return () => clearTimeout(timer)
  }, [messages, isLoading])

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex-1 min-h-0">
        <ChatEmptyState />
      </div>
    )
  }

  return (
    <ScrollArea className="flex-1 min-h-0 h-full px-3 py-4 md:px-6" ref={scrollRef}>
      <div className="max-w-4xl mx-auto flex flex-col gap-6 pb-4">
        {messages.map((msg, index) => (
          <div key={`${msg.id || 'msg'}-${index}`} className="w-full">
            <ChatMessage
              message={msg}
              citations={
                msg.citations && msg.citations.length > 0
                  ? msg.citations
                  : index === messages.length - 1 && msg.role === 'assistant'
                    ? latestCitations
                    : undefined
              }
              onEdit={onEditMessage}
              onRegenerate={onRegenerateMessage}
              isSending={isLoading}
              disableRegenerate={
                msg.role === 'assistant' &&
                index > 0 &&
                Boolean(
                  messages[index - 1].localImageUrl ||
                  (messages[index - 1].attachments && messages[index - 1].attachments!.length > 0 && messages[index - 1].attachments!.some(a => a.mime_type.startsWith('image/')))
                )
              }
            />
          </div>
        ))}
        {isLoading && (
          <div className="w-full pt-1">
            <TypingIndicator label={loadingLabel} />
          </div>
        )}
        {!isLoading && messages.length > 0 && messages[messages.length - 1].role === 'assistant' && (
          <div className="flex flex-wrap gap-2 mt-2 pt-2 md:ml-11 animate-in fade-in-0 duration-500">
            {['Can you summarize this?', 'Explain this simply', 'What are the main takeaways?'].map((q) => (
              <Button
                key={q}
                variant="outline"
                size="sm"
                className="rounded-full text-xs text-muted-foreground hover:text-foreground bg-background/50 h-7"
                onClick={() => onSuggestedClick?.(q)}
              >
                {q}
              </Button>
            ))}
          </div>
        )}
        <div ref={bottomRef} className="h-px w-full" />
      </div>
    </ScrollArea>
  )
}
