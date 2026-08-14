import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, User, Copy, Check, X, Maximize2, FileText, Pencil, RefreshCw, Cpu } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Message, Citation } from '../types/chat'
import { CitationCard } from './CitationCard'
import { Button } from '@/components/ui/button'

interface ChatMessageProps {
  message: Message
  citations?: Citation[]
  onEdit?: (message: Message) => void
  onRegenerate?: (message: Message) => void
  isSending?: boolean
}

export function ChatMessage({
  message,
  citations,
  onEdit,
  onRegenerate,
  isSending,
}: ChatMessageProps) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)
  const [isLightboxOpen, setIsLightboxOpen] = useState(false)

  const handleCopy = () => {
    if (!message.content) return
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Handle ESC key for lightbox modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isLightboxOpen) {
        setIsLightboxOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isLightboxOpen])

  // Resolve image source: local object URL or attachment metadata
  const imageAttachment = message.attachments?.find((att) =>
    att.mime_type.startsWith('image/'),
  )

  return (
    <>
      <div
        className={cn(
          'flex w-full gap-3 group relative transition-all duration-150',
          isUser ? 'justify-end' : 'justify-start',
        )}
      >
        {!isUser && (
          <div className="shrink-0 mt-0.5">
            <div className="w-8 h-8 rounded-xl bg-primary/15 text-primary border border-primary/20 flex items-center justify-center shadow-xs">
              <Bot className="w-4 h-4" />
            </div>
          </div>
        )}

        <div
          className={cn(
            'flex flex-col min-w-0 max-w-[85%] md:max-w-[75%]',
            isUser ? 'items-end' : 'items-start flex-1',
          )}
        >
          <div
            className={cn(
              'rounded-2xl px-4 py-3 shadow-xs space-y-3 transition-all duration-150',
              isUser
                ? 'bg-primary/10 border border-primary/20 text-foreground rounded-tr-xs'
                : 'bg-card/70 border border-border/60 text-card-foreground rounded-tl-xs w-full',
            )}
          >
            {/* User or Assistant message content */}
            {message.content && (
              <div
                className={cn(
                  'prose prose-sm dark:prose-invert max-w-none wrap-break-word leading-relaxed text-sm',
                  isUser ? 'text-foreground font-normal' : 'text-foreground/90',
                )}
              >
                {isUser ? (
                  <p className="whitespace-pre-wrap m-0">{message.content}</p>
                ) : (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => (
                        <p className="mb-2.5 last:mb-0 leading-relaxed text-foreground/90">
                          {children}
                        </p>
                      ),
                      ul: ({ children }) => (
                        <ul className="list-disc list-inside space-y-1 my-2 text-foreground/90 pl-1">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="list-decimal list-inside space-y-1 my-2 text-foreground/90 pl-1">
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => (
                        <li className="text-foreground/90 leading-snug">{children}</li>
                      ),
                      h1: ({ children }) => (
                        <h1 className="text-base font-semibold text-foreground mt-3 mb-1.5">
                          {children}
                        </h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-sm font-semibold text-foreground mt-2.5 mb-1">
                          {children}
                        </h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-xs font-semibold text-foreground mt-2 mb-1 uppercase tracking-wider">
                          {children}
                        </h3>
                      ),
                      code: ({ children }) => (
                        <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono border border-border/40 text-primary">
                          {children}
                        </code>
                      ),
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-2 border-primary/40 pl-3 italic text-muted-foreground my-2">
                          {children}
                        </blockquote>
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                )}
              </div>
            )}

            {/* Integrated Image Preview inside user message bubble */}
            {(message.localImageUrl || imageAttachment) && (
              <div className="pt-1">
                <div
                  onClick={() => setIsLightboxOpen(true)}
                  className="group/img relative inline-block overflow-hidden rounded-xl border border-border/50 bg-background/50 cursor-pointer shadow-xs hover:border-primary/40 transition-all duration-200"
                  title="Click to expand image"
                >
                  {message.localImageUrl ? (
                    <img
                      src={message.localImageUrl}
                      alt="User attached file"
                      className="max-h-64 sm:max-h-72 w-auto max-w-full object-contain rounded-lg transition-transform duration-200 group-hover/img:scale-[1.01]"
                    />
                  ) : imageAttachment ? (
                    <div className="flex items-center gap-2 p-2.5 bg-muted/30 max-w-xs text-xs rounded-lg">
                      <FileText className="h-4 w-4 text-primary shrink-0" />
                      <span className="truncate font-medium">{imageAttachment.filename}</span>
                      <span className="text-muted-foreground shrink-0 text-[10px]">
                        ({(imageAttachment.size / 1024).toFixed(1)} KB)
                      </span>
                    </div>
                  ) : null}

                  {message.localImageUrl && (
                    <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/20 transition-colors flex items-center justify-center">
                      <div className="opacity-0 group-hover/img:opacity-100 transition-opacity bg-black/70 text-white p-1.5 rounded-full backdrop-blur-xs shadow-md">
                        <Maximize2 className="h-4 w-4" />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Non-image attachment chip */}
            {message.attachments &&
              message.attachments.length > 0 &&
              !message.localImageUrl &&
              !imageAttachment && (
                <div className="flex flex-col gap-1.5 pt-1">
                  {message.attachments.map((att) => (
                    <div
                      key={att.id}
                      className="flex items-center gap-2 p-2 rounded-lg border border-border/50 bg-muted/20 max-w-xs text-xs"
                    >
                      <div className="h-7 w-7 rounded-md bg-primary/10 flex items-center justify-center text-primary shrink-0 font-semibold text-[10px] uppercase">
                        {att.mime_type.split('/')[1] || 'FILE'}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="font-medium truncate" title={att.filename}>
                          {att.filename}
                        </p>
                        <p className="text-muted-foreground text-[10px]">
                          {(att.size / 1024).toFixed(1)} KB
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

            {/* Citations section for Assistant responses */}
            {citations && citations.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border/50 space-y-2">
                <p className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                  <span>Sources</span>
                  <span className="text-[10px] bg-muted px-1.5 py-0.5 rounded-full font-mono">
                    {citations.length}
                  </span>
                </p>
                <div className="grid gap-2 grid-cols-1 md:grid-cols-2">
                  {citations.map((citation, index) => (
                    <CitationCard key={`${citation.chunk_id}-${index}`} citation={citation} />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Action Bar & Model Used Badge */}
          <div
            className={cn(
              'flex items-center gap-1.5 mt-1.5 transition-opacity duration-150',
              isUser ? 'flex-row-reverse opacity-90 sm:opacity-0 group-hover:opacity-100' : 'flex-row opacity-90 sm:opacity-0 group-hover:opacity-100',
            )}
          >
            {/* User Actions */}
            {isUser && onEdit && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground rounded-lg gap-1 border border-transparent hover:border-border/40"
                onClick={() => onEdit(message)}
                disabled={isSending}
                title="Edit message"
                aria-label="Edit message"
              >
                <Pencil className="h-3 w-3" />
                <span>Edit</span>
              </Button>
            )}

            {/* Assistant Actions */}
            {!isUser && onRegenerate && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground rounded-lg gap-1 border border-transparent hover:border-border/40"
                onClick={() => onRegenerate(message)}
                disabled={isSending}
                title="Regenerate response"
                aria-label="Regenerate response"
              >
                <RefreshCw className={cn("h-3 w-3", isSending && "animate-spin")} />
                <span>Regenerate</span>
              </Button>
            )}

            {/* Common Copy Action */}
            {message.content && (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 text-muted-foreground hover:text-foreground rounded-lg"
                onClick={handleCopy}
                title="Copy text"
                aria-label="Copy text"
              >
                {copied ? (
                  <Check className="h-3.5 w-3.5 text-green-500" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </Button>
            )}

            {/* Per-message Model Information Badge (Assistant only) */}
            {!isUser && message.model_used && (
              <span
                className="text-[10px] font-mono text-muted-foreground/70 bg-muted/30 px-1.5 py-0.5 rounded border border-border/30 ml-1 inline-flex items-center gap-1"
                title={`Model used: ${message.model_used}`}
              >
                <Cpu className="h-2.5 w-2.5 shrink-0 opacity-60" />
                {message.model_used}
              </span>
            )}
          </div>
        </div>

        {isUser && (
          <div className="shrink-0 mt-0.5">
            <div className="w-8 h-8 rounded-xl bg-primary text-primary-foreground flex items-center justify-center shadow-xs">
              <User className="w-4 h-4" />
            </div>
          </div>
        )}
      </div>

      {/* Lightbox Modal for enlarged image preview */}
      {isLightboxOpen && message.localImageUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-xs flex items-center justify-center p-4 animate-in fade-in-0"
          onClick={() => setIsLightboxOpen(false)}
        >
          <div className="relative max-w-4xl max-h-[90vh] flex flex-col items-center">
            <button
              type="button"
              onClick={() => setIsLightboxOpen(false)}
              className="absolute -top-10 right-0 p-2 text-white/80 hover:text-white bg-white/10 hover:bg-white/20 rounded-full transition"
              title="Close full view (Esc)"
              aria-label="Close image preview"
            >
              <X className="h-5 w-5" />
            </button>
            <img
              src={message.localImageUrl}
              alt="Enlarged preview"
              className="max-h-[85vh] max-w-[90vw] object-contain rounded-xl shadow-2xl border border-white/10"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </>
  )
}
