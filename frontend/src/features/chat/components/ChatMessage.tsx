import { useState, useEffect, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, User, Copy, Check, X, FileText, Globe, Layers, Pencil, RefreshCw, Cpu } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Message, Citation } from '../types/chat'
import { Button } from '@/components/ui/button'
import { AttachmentCard } from './AttachmentCard'

interface ChatMessageProps {
  message: Message
  citations?: Citation[]
  onEdit?: (message: Message) => void
  onRegenerate?: (message: Message) => void
  isSending?: boolean
  disableRegenerate?: boolean
}

// Extract string text from React children to check for citation-only blocks
function getTextFromChildren(children: React.ReactNode): string {
  if (typeof children === 'string') return children
  if (typeof children === 'number') return String(children)
  if (Array.isArray(children)) return children.map(getTextFromChildren).join('')
  if (children && typeof children === 'object' && 'props' in (children as any)) {
    return getTextFromChildren((children as any).props?.children)
  }
  return ''
}

function isCitationOnlyParagraph(children: React.ReactNode): boolean {
  const text = getTextFromChildren(children).trim()
  if (!text) return false
  const stripped = text
    .replace(/\[(?:[^\[\]\n]+\.(?:docx?|pdf|txt|md|xlsx?|pptx?|csv)[^\[\]\n]*|C\d+)\]/gi, '')
    .replace(/[,\s]+/g, '')
    .trim()
  return stripped.length === 0
}

// Helper to render inline RAG citation chips with clean alignment
function renderInlineCitations(text: string, citations: Citation[] = []) {
  const parts: (string | React.ReactElement)[] = []
  const citationRegex = /\[((?:[^\[\]\n]+\.(?:docx?|pdf|txt|md|xlsx?|pptx?|csv)[^\[\]\n]*)|(?:C\d+))\]/gi
  let lastIdx = 0
  let match: RegExpExecArray | null

  while ((match = citationRegex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.substring(lastIdx, match.index))
    }
    const rawTag = match[1].trim()
    const matchedCit = citations.find(c => {
      if (c.citation_label && c.citation_label.includes(rawTag)) return true
      if (c.citation_id && c.citation_id.toLowerCase() === rawTag.toLowerCase()) return true
      if (c.document_title && rawTag.toLowerCase().includes(c.document_title.toLowerCase())) return true
      if (c.file_name && rawTag.toLowerCase().includes(c.file_name.toLowerCase())) return true
      return false
    })

    const tooltip = matchedCit?.chunk_text
      ? `${matchedCit.document_title || matchedCit.file_name || 'Document'}${matchedCit.source_location ? ` • ${matchedCit.source_location}` : ''}\n\n"${matchedCit.chunk_text.slice(0, 200)}..."`
      : rawTag

    const partsTag = rawTag.split(',')
    const docPart = partsTag[0].trim()
    const locPart = partsTag.slice(1).join(',').trim()

    // Clean up location display for sleek badge (extract leaf section if breadcrumbs exist)
    let cleanLoc = locPart
    for (const sep of ['→', '->', '>']) {
      if (cleanLoc.includes(sep)) {
        cleanLoc = cleanLoc.split(sep).pop()?.trim() || cleanLoc
      }
    }
    cleanLoc = cleanLoc.replace(/^Section:\s*/i, '')

    parts.push(
      <span
        key={`cit-${match.index}`}
        className="group/cit inline-flex items-center gap-1.5 text-[11px] leading-none font-medium text-primary bg-primary/10 hover:bg-primary/15 transition-all rounded-md px-2 py-1 mx-0.5 border border-primary/20 cursor-default select-none align-middle shadow-2xs"
        title={tooltip}
      >
        <FileText className="w-3.5 h-3.5 text-primary/70 shrink-0" />
        <span className="font-medium tracking-normal text-foreground/90 max-w-50 sm:max-w-75 truncate">
          {docPart}
        </span>
        {cleanLoc && (
          <span className="text-[10px] text-muted-foreground/80 pl-1.5 border-l border-primary/25 max-w-45 sm:max-w-60 truncate font-normal">
            {cleanLoc}
          </span>
        )}
      </span>
    )
    lastIdx = match.index + match[0].length
  }

  if (lastIdx === 0) {
    return text
  }

  if (lastIdx < text.length) {
    parts.push(text.substring(lastIdx))
  }

  return parts
}

function processChildrenWithCitations(children: React.ReactNode, citations: Citation[] = []): React.ReactNode {
  if (typeof children === 'string') {
    return renderInlineCitations(children, citations)
  }
  if (Array.isArray(children)) {
    return children.map((child, idx) => {
      if (typeof child === 'string') {
        return <span key={idx}>{renderInlineCitations(child, citations)}</span>
      }
      return child
    })
  }
  return children
}

export function ChatMessage({
  message,
  citations: _citations,
  onEdit,
  onRegenerate,
  isSending,
  disableRegenerate,
}: ChatMessageProps) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)
  const [isLightboxOpen, setIsLightboxOpen] = useState(false)
  const activeCitations = _citations || (message as any).citations || (message as any).sources || []

  const handleCopy = () => {
    if (!message.content) return
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Preprocess markdown to ensure citations at the end of lists are detached into their own block
  // so they don't get trapped or awkwardly indented inside <li> tags
  const formattedContent = useMemo(() => {
    if (!message.content) return ''
    let text = message.content

    // 1. If a list item ends with a citation tag (e.g. "- Item text [Doc.docx...]"),
    // split the citation into its own block so it doesn't get trapped inside the <li>
    text = text.replace(
      /((?:^|\n)\s*(?:[-*•]|\d+\.)\s+[^\n]+?)\s*(\[(?:[^\[\]\n]+\.(?:docx?|pdf|txt|md|xlsx?|pptx?|csv)[^\[\]\n]*|C\d+)\])\s*$/gim,
      '$1\n\n$2'
    )

    // 2. If a citation is immediately on the line below a list item without a blank line
    // e.g. "- Item text\n[Doc.docx...]" -> "- Item text\n\n[Doc.docx...]"
    text = text.replace(
      /((?:^|\n)\s*(?:[-*•]|\d+\.)\s+[^\n]+)\n(\s*\[(?:[^\[\]\n]+\.(?:docx?|pdf|txt|md|xlsx?|pptx?|csv)[^\[\]\n]*|C\d+)\]\s*)/gim,
      '$1\n\n$2'
    )

    return text
  }, [message.content])

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
              'space-y-3 transition-all duration-150',
              isUser
                ? 'bg-primary/8 border border-primary/20 text-foreground rounded-2xl rounded-tr-xs px-4 py-2.5 shadow-xs max-w-fit'
                : 'bg-transparent border-0 text-card-foreground px-0 py-1 w-full',
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
                      p: ({ children }) => {
                        if (isCitationOnlyParagraph(children)) {
                          return (
                            <div className="mt-3.5 pt-2 border-t border-border/40 flex items-center flex-wrap gap-2 text-xs text-muted-foreground w-full not-prose">
                              <span className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground/70 uppercase tracking-wider select-none">
                                Source:
                              </span>
                              {processChildrenWithCitations(children, activeCitations)}
                            </div>
                          )
                        }
                        return (
                          <p className="mb-3 last:mb-0 leading-relaxed text-sm text-foreground/90 font-sans">
                            {processChildrenWithCitations(children, activeCitations)}
                          </p>
                        )
                      },
                      ul: ({ children }) => (
                        <ul className="list-disc pl-5 space-y-1.5 my-3 text-sm text-foreground/90">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="list-decimal pl-5 space-y-1.5 my-3 text-sm text-foreground/90">
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => (
                        <li className="text-foreground/90 leading-relaxed pl-0.5">{processChildrenWithCitations(children, activeCitations)}</li>
                      ),
                      h1: ({ children }) => (
                        <h1 className="text-base font-bold text-foreground mt-4 mb-2 font-display">
                          {children}
                        </h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-sm font-semibold text-foreground mt-3.5 mb-1.5 font-display">
                          {children}
                        </h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-xs font-semibold text-foreground mt-3 mb-1 uppercase tracking-wider font-display">
                          {children}
                        </h3>
                      ),
                      code: ({ className, children, ...props }) => {
                        const match = /language-(\w+)/.exec(className || '')
                        return match ? (
                          <pre className="bg-muted/40 border border-border/30 rounded-lg p-3 my-3 overflow-x-auto text-xs font-mono leading-relaxed text-foreground scrollbar-thin">
                            <code className={className} {...props}>
                              {children}
                            </code>
                          </pre>
                        ) : (
                          <code className="bg-muted/70 px-1.5 py-0.5 rounded text-xs font-mono border border-border/20 text-primary" {...props}>
                            {children}
                          </code>
                        )
                      },
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-2 border-primary/30 pl-3 italic text-muted-foreground/80 my-3">
                          {processChildrenWithCitations(children, activeCitations)}
                        </blockquote>
                      ),
                    }}
                  >
                    {formattedContent}
                  </ReactMarkdown>
                )}
              </div>
            )}

            {/* Attachments rendering using AttachmentCard */}
            {((message.attachments && message.attachments.length > 0) || message.localImageUrl) && (
              <div className="flex flex-wrap gap-2 pt-1.5 max-w-full">
                {message.attachments && message.attachments.length > 0 ? (
                  message.attachments.map((att, idx) => (
                    <AttachmentCard
                      key={att.id || (att as any).storage_path || (att as any).filename || idx}
                      attachment={{
                        ...att,
                        previewUrl: att.previewUrl || ((att.mime_type || (att as any).mimeType || '')?.startsWith('image/') ? message.localImageUrl : undefined),
                      }}
                      readOnly
                    />
                  ))
                ) : message.localImageUrl ? (
                  <AttachmentCard
                    attachment={{
                      id: 'local-img',
                      filename: 'Attached Image',
                      mime_type: 'image/png',
                      size: 0,
                      previewUrl: message.localImageUrl,
                    }}
                    readOnly
                  />
                ) : null}
              </div>
            )}

          </div>

          {/* Action Bar & Model Used Badge */}
          <div
            className={cn(
              'flex items-center gap-1 mt-2 transition-all duration-200',
              isUser ? 'flex-row-reverse opacity-100 sm:opacity-0 focus-within:opacity-100 group-hover:opacity-100 -mr-2' : 'flex-row opacity-100 sm:opacity-0 focus-within:opacity-100 group-hover:opacity-100 -ml-2',
            )}
          >
            {/* User Actions */}
            {isUser && onEdit && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/60 rounded-lg gap-2 transition-colors"
                onClick={() => onEdit(message)}
                disabled={isSending}
                title="Edit message"
                aria-label="Edit message"
              >
                <Pencil className="h-3.5 w-3.5" />
                <span>Edit</span>
              </Button>
            )}

            {/* Assistant Actions */}
            {!isUser && onRegenerate && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/60 rounded-lg gap-2 transition-colors"
                onClick={() => onRegenerate(message)}
                disabled={isSending}
                title={disableRegenerate ? "Cannot regenerate queries containing images" : "Regenerate response"}
                aria-label="Regenerate response"
              >
                <RefreshCw className={cn("h-3.5 w-3.5", isSending && "animate-spin", disableRegenerate && "opacity-50")} />
                <span className={cn(disableRegenerate && "opacity-50 line-through")}>Regenerate</span>
              </Button>
            )}

            {/* Common Copy Action */}
            {message.content && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/60 rounded-lg gap-2 transition-colors"
                onClick={handleCopy}
                title="Copy text"
                aria-label="Copy text"
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-emerald-500" />
                    <span className="text-emerald-500">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" />
                    <span>Copy</span>
                  </>
                )}
              </Button>
            )}

            {/* Per-message Model Information & Telemetry (Assistant only) */}
            {!isUser && message.model_used && (
              <span
                className="text-[10px] font-mono text-muted-foreground/60 bg-muted/40 px-2 py-1 rounded-md border border-border/40 ml-1 inline-flex items-center gap-2 cursor-default transition-colors hover:bg-muted/60 hover:text-muted-foreground"
                title={`Generated with ${message.model_used}`}
              >
                <span className="flex items-center gap-1.5 border-r border-border/40 pr-2">
                  <Cpu className="h-3 w-3 shrink-0 opacity-70" />
                  {message.model_used}
                </span>
                
                {message.latency_ms ? (
                  <span className="flex items-center gap-2">
                    <span title="Total latency">{(message.latency_ms / 1000).toFixed(1)}s</span>
                    
                    {message.total_tokens ? (
                      <>
                        <span title="Total tokens">{message.total_tokens}t</span>
                        <span className="opacity-75" title="Tokens per second">
                          {((message.total_tokens / message.latency_ms) * 1000).toFixed(1)} t/s
                        </span>
                      </>
                    ) : null}
                  </span>
                ) : null}
              </span>
            )}

            {/* Retrieval Mode Badge (Assistant only) */}
            {!isUser && message.retrieval_mode && (
              <span
                className={cn(
                  "text-[10px] font-medium px-2 py-1 rounded-md border inline-flex items-center gap-1.5 cursor-default transition-colors",
                  message.retrieval_mode === 'web'
                    ? "bg-blue-500/10 text-blue-500 border-blue-500/20"
                    : message.retrieval_mode === 'hybrid'
                    ? "bg-purple-500/10 text-purple-500 border-purple-500/20"
                    : "bg-muted/40 text-muted-foreground border-border/40"
                )}
                title={`Retrieval Mode: ${message.retrieval_mode}`}
              >
                {message.retrieval_mode === 'web' ? (
                  <>
                    <Globe className="h-3 w-3 shrink-0" />
                    <span>Web Search</span>
                  </>
                ) : message.retrieval_mode === 'hybrid' ? (
                  <>
                    <Layers className="h-3 w-3 shrink-0" />
                    <span>Hybrid Search</span>
                  </>
                ) : (
                  <>
                    <FileText className="h-3 w-3 shrink-0" />
                    <span>Local RAG</span>
                  </>
                )}
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
