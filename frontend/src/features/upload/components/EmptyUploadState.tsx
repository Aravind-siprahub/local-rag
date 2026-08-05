import { FolderUpIcon } from 'lucide-react'

export function EmptyUploadState() {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center rounded-xl border border-dashed border-border/60 bg-muted/20">
      <div className="p-3 rounded-full bg-muted/60 text-muted-foreground mb-3">
        <FolderUpIcon className="size-6" />
      </div>
      <h4 className="text-sm font-medium text-foreground">Upload Queue Empty</h4>
      <p className="text-xs text-muted-foreground mt-1 max-w-sm">
        Drag and drop files into the box above or click browse to add documents to your ingestion queue.
      </p>
    </div>
  )
}
