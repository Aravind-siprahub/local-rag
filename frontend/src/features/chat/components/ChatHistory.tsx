import { useEffect, useRef } from 'react'
import { ChatMessage } from './ChatMessage'
import { TypingIndicator } from './TypingIndicator'
import { ChatEmptyState } from './EmptyState'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { Message, Citation } from '../types/chat'

interface ChatHistoryProps {
  messages: Message[]
  isLoading?: boolean
  latestCitations?: Citation[]
  loadingLabel?: string
  onEditMessage?: (message: Message) => void
  onRegenerateMessage?: (message: Message) => void
}

export function ChatHistory({
  messages,
  isLoading,
  latestCitations,
  loadingLabel,
  onEditMessage,
  onRegenerateMessage,
}: ChatHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      const viewport =
        scrollRef.current.querySelector('[data-radix-scroll-area-viewport]') || scrollRef.current
      viewport.scrollTop = viewport.scrollHeight
    }
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
      <div className="max-w-3xl mx-auto flex flex-col gap-5 pb-4">
        {messages.map((msg, index) => (
          <div key={msg.id || index} className="w-full">
            <ChatMessage
              message={msg}
              citations={
                index === messages.length - 1 && msg.role === 'assistant'
                  ? latestCitations
                  : undefined
              }
              onEdit={onEditMessage}
              onRegenerate={onRegenerateMessage}
              isSending={isLoading}
            />
          </div>
        ))}
        {isLoading && (
          <div className="w-full pt-1">
            <TypingIndicator label={loadingLabel} />
          </div>
        )}
      </div>
    </ScrollArea>
  )
}
