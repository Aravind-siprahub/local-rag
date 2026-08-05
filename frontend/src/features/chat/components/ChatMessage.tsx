import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, User, Copy, Check } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'
import type { Message, Citation } from '../types/chat'
import { CitationCard } from './CitationCard'
import { Button } from '@/components/ui/button'

interface ChatMessageProps {
  message: Message
  citations?: Citation[]
}

export function ChatMessage({ message, citations }: ChatMessageProps) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={cn("flex w-full gap-4 p-4 rounded-lg", isUser ? "bg-background" : "bg-muted/30")}>
      <div className="shrink-0 mt-1">
        {isUser ? (
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground">
            <User className="w-5 h-5" />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center text-secondary-foreground border">
            <Bot className="w-5 h-5" />
          </div>
        )}
      </div>
      
      <div className="flex-1 space-y-4 min-w-0">
        <div className="prose prose-sm md:prose-base dark:prose-invert max-w-none wrap-break-word">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>
        
        {citations && citations.length > 0 && (
          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-sm font-medium mb-2 text-muted-foreground">Sources</p>
            <div className="grid gap-2 grid-cols-1 md:grid-cols-2">
              {citations.map((citation, index) => (
                <CitationCard key={`${citation.chunk_id}-${index}`} citation={citation} />
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
        <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground" onClick={handleCopy} title="Copy message">
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
        </Button>
      </div>
    </div>
  )
}
