import { Badge } from '@/components/ui/badge'

export function UploadHeader() {
  return (
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-border/40">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Upload Documents</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Ingest raw documents into your local vector database for instant RAG search and retrieval.
        </p>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-medium text-muted-foreground mr-1">Supported formats:</span>
        <Badge variant="outline" className="font-mono text-xs bg-muted/30">
          PDF
        </Badge>
        <Badge variant="outline" className="font-mono text-xs bg-muted/30">
          DOCX
        </Badge>
        <Badge variant="outline" className="font-mono text-xs bg-muted/30">
          TXT
        </Badge>
        <Badge variant="outline" className="font-mono text-xs bg-muted/30">
          MD
        </Badge>
      </div>
    </div>
  )
}
