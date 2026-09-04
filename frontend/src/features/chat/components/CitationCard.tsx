import { useState } from 'react'
import { FileText, Globe, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react'
import type { Citation } from '../types/chat'
import { cn } from '@/lib/utils'

interface CitationCardProps {
  citation: Citation
}

export function CitationCard({ citation }: CitationCardProps) {
  const [expanded, setExpanded] = useState(false)
  const isWeb = citation.source_type === 'web' || Boolean(citation.url)
  const docTitle = citation.document_title || (isWeb ? "Web Source" : "Source Document")
  const domain = citation.domain || (citation.url ? new URL(citation.url).hostname.replace("www.", "") : null)

  const locationInfo = isWeb
    ? domain
    : [
        citation.page_number ? `Page ${citation.page_number}` : null,
        citation.section_title ? citation.section_title : null,
      ].filter(Boolean).join(" • ")

  return (
    <div className="group mt-2 border border-border/60 rounded-xl p-3 bg-card/50 text-card-foreground text-sm flex flex-col gap-2.5 shadow-xs transition-all duration-200 hover:border-primary/30 hover:bg-primary/5">
      <div 
        className="flex items-start justify-between gap-3 cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <div className={cn(
            "w-7 h-7 rounded-lg flex items-center justify-center shrink-0",
            isWeb ? "bg-blue-500/10 text-blue-500" : "bg-primary/10 text-primary"
          )}>
            {isWeb ? <Globe className="w-3.5 h-3.5" /> : <FileText className="w-3.5 h-3.5" />}
          </div>
          <div className="flex flex-col min-w-0">
            <span className="truncate max-w-50 sm:max-w-62.5 font-medium text-foreground text-[13px] leading-tight">
              {docTitle}
            </span>
            {locationInfo && (
              <span className="text-[10px] text-muted-foreground mt-0.5 truncate">
                {locationInfo}
              </span>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-1.5 shrink-0 text-muted-foreground">
          {citation.url ? (
            <a
              href={citation.url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="flex items-center gap-1 text-[11px] font-medium text-blue-500 hover:text-blue-600 bg-blue-500/10 px-2 py-0.5 rounded-md hover:bg-blue-500/20 transition-colors"
              title="Open web source"
            >
              <span>Visit</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          ) : (
            <div className="text-[10px] bg-muted/60 px-1.5 py-0.5 rounded text-muted-foreground/80 font-mono">
              {Math.min(100, Math.max(0, Math.round(citation.similarity_score * 100)))}% match
            </div>
          )}
          <button 
            className="p-1 rounded-md hover:bg-muted/80 hover:text-foreground transition-colors"
            aria-label={expanded ? "Collapse citation" : "Expand citation"}
          >
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>
      
      {citation.chunk_text && (
        <div 
          className={cn(
            "text-muted-foreground/90 text-xs italic border-l-[3px] border-primary/20 pl-3 py-1 transition-all duration-200 overflow-hidden",
            expanded ? "max-h-125" : "max-h-15 line-clamp-3"
          )}
        >
          "{citation.chunk_text}"
        </div>
      )}
    </div>
  )
}

