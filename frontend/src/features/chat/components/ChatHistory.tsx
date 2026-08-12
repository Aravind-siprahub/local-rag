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
}

export function ChatHistory({ messages, isLoading, latestCitations, loadingLabel }: ChatHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      const viewport = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]') || scrollRef.current
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
    <ScrollArea className="flex-1 min-h-0 h-full px-4 py-4" ref={scrollRef}>
      <div className="max-w-3xl mx-auto flex flex-col gap-6 pb-6">
        {messages.map((msg, index) => (
          <div key={msg.id || index} className="group">
            <ChatMessage 
              message={msg} 
              citations={
                // Only attach citations to the very last assistant message if we have them in state
                (index === messages.length - 1 && msg.role === 'assistant') ? latestCitations : undefined
              } 
            />
          </div>
        ))}
        {isLoading && (
          <div className="max-w-3xl mx-auto w-full group">
            <TypingIndicator label={loadingLabel} />
          </div>
        )}
      </div>
    </ScrollArea>
  )
}
