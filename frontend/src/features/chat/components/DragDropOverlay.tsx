import { UploadCloud } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DragDropOverlayProps {
  isDragging: boolean
}

export function DragDropOverlay({ isDragging }: DragDropOverlayProps) {
  if (!isDragging) return null

  return (
    <div
      className={cn(
        'absolute inset-0 z-50 flex flex-col items-center justify-center gap-2 rounded-xl',
        'border-2 border-dashed border-primary bg-primary/10 backdrop-blur-xs',
        'animate-in fade-in-0 zoom-in-95 duration-150 text-primary pointer-events-none',
      )}
    >
      <div className="p-3 rounded-full bg-primary/20 animate-bounce">
        <UploadCloud className="h-7 w-7 text-primary" />
      </div>
      <div className="flex flex-col items-center">
        <span className="text-sm font-semibold text-foreground">Drop files to attach</span>
        <span className="text-xs text-muted-foreground">PDF, DOCX, CSV, XLSX, PPTX, TXT, PNG, JPEG, WEBP</span>
      </div>
    </div>
  )
}
