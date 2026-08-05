import { FileText } from 'lucide-react'
import type { Citation } from '../types/chat'

interface CitationCardProps {
  citation: Citation
}

export function CitationCard({ citation }: CitationCardProps) {
  return (
    <div className="mt-2 border rounded-md p-3 bg-card text-card-foreground text-sm flex flex-col gap-2 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 font-medium text-primary">
          <FileText className="w-4 h-4" />
          <span>Source Document</span>
        </div>
        <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
          Similarity: {(citation.similarity_score * 100).toFixed(0)}%
        </span>
      </div>
      <p className="text-muted-foreground text-xs line-clamp-3 italic border-l-2 pl-2">
        "{citation.chunk_text}"
      </p>
    </div>
  )
}
