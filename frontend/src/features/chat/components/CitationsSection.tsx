import type { Citation } from '../types/chat'
import { CitationCard } from './CitationCard'

interface CitationsSectionProps {
  citations?: Citation[]
}

export function CitationsSection({ citations }: CitationsSectionProps) {
  if (!citations || citations.length === 0) {
    return null
  }

  return (
    <div className="mt-3 pt-2 border-t border-border/40 flex flex-col gap-1.5 w-full">
      <div className="text-[11px] font-semibold tracking-wider text-muted-foreground/80 uppercase px-0.5">
        Sources & Citations ({citations.length})
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {citations.map((citation, idx) => (
          <CitationCard
            key={citation.chunk_id || citation.url || idx}
            citation={citation}
          />
        ))}
      </div>
    </div>
  )
}

