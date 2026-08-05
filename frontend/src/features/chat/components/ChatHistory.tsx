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
}

export function ChatHistory({ messages, isLoading, latestCitations }: ChatHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      const scrollElement = scrollRef.current
      scrollElement.scrollTop = scrollElement.scrollHeight
    }
  }, [messages, isLoading])

  if (messages.length === 0 && !isLoading) {
    return <ChatEmptyState />
  }

  return (
    <ScrollArea className="flex-1 px-4 py-4" ref={scrollRef}>
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
            <TypingIndicator />
          </div>
        )}
      </div>
    </ScrollArea>
  )
}
