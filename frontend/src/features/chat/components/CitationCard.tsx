import { useState } from 'react'
import { FileText, ChevronDown, ChevronUp } from 'lucide-react'
import type { Citation } from '../types/chat'
import { cn } from '@/lib/utils'

interface CitationCardProps {
  citation: Citation
}

export function CitationCard({ citation }: CitationCardProps) {
  const [expanded, setExpanded] = useState(false)
  const docTitle = citation.document_title || "Source Document"
  const locationInfo = [
    citation.page_number ? `Page ${citation.page_number}` : null,
    citation.section_title ? citation.section_title : null,
  ].filter(Boolean).join(" • ")

  return (
    <div className="group mt-2 border border-border/60 rounded-xl p-3 bg-card/50 text-card-foreground text-sm flex flex-col gap-2.5 shadow-xs transition-all duration-200 hover:border-primary/30 hover:bg-primary/2">
      <div 
        className="flex items-start justify-between gap-3 cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-primary/10 text-primary flex items-center justify-center shrink-0">
            <FileText className="w-3.5 h-3.5" />
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
          <div className="text-[10px] bg-muted/60 px-1.5 py-0.5 rounded text-muted-foreground/80 font-mono">
            {(citation.similarity_score * 100).toFixed(0)}% match
          </div>
          <button 
            className="p-1 rounded-md hover:bg-muted/80 hover:text-foreground transition-colors"
            aria-label={expanded ? "Collapse citation" : "Expand citation"}
          >
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>
      
      <div 
        className={cn(
          "text-muted-foreground/90 text-xs italic border-l-[3px] border-primary/20 pl-3 py-1 transition-all duration-200 overflow-hidden",
          expanded ? "max-h-125" : "max-h-15 line-clamp-3"
        )}
      >
        "{citation.chunk_text}"
      </div>
    </div>
  )
}
