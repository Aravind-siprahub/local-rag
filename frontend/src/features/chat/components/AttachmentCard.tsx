import { FileText, FileSpreadsheet, Presentation, FileCode, File, X, AlertCircle } from 'lucide-react'
import type { Attachment } from '../types/chat'
import { cn } from '@/lib/utils'

interface AttachmentCardProps {
  attachment: Attachment
  onRemove?: (id: string) => void
  readOnly?: boolean
}

export function AttachmentCard({ attachment, onRemove, readOnly = false }: AttachmentCardProps) {
  const filename = attachment.filename || (attachment as any).name || 'File'
  const mimeType = attachment.mime_type || (attachment as any).mimeType || ''

  const isImage = mimeType.startsWith('image/') || 
                  ['png', 'jpg', 'jpeg', 'webp'].some(ext => filename.toLowerCase().endsWith(`.${ext}`))

  const formatSize = (bytes: number) => {
    if (!bytes) return ''
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getFileIcon = (fname: string, mime: string) => {
    const lowerName = fname.toLowerCase()
    if (mime.includes('csv') || lowerName.endsWith('.csv')) return <FileSpreadsheet className="h-5 w-5 text-emerald-500" />
    if (mime.includes('excel') || mime.includes('spreadsheet') || lowerName.endsWith('.xlsx') || lowerName.endsWith('.xls')) {
      return <FileSpreadsheet className="h-5 w-5 text-green-500" />
    }
    if (mime.includes('pdf') || lowerName.endsWith('.pdf')) return <FileText className="h-5 w-5 text-rose-500" />
    if (mime.includes('word') || lowerName.endsWith('.docx') || lowerName.endsWith('.doc')) return <FileText className="h-5 w-5 text-blue-500" />
    if (mime.includes('presentation') || lowerName.endsWith('.pptx') || lowerName.endsWith('.ppt')) return <Presentation className="h-5 w-5 text-amber-500" />
    if (lowerName.endsWith('.txt') || lowerName.endsWith('.md') || lowerName.endsWith('.json')) return <FileCode className="h-5 w-5 text-slate-400" />
    return <File className="h-5 w-5 text-indigo-400" />
  }

  const storagePath = (attachment as any).storage_path
  const imageUrl = attachment.previewUrl || attachment.url || 
    (storagePath ? `https://bwtzzohvfcscfuyyeifr.supabase.co/storage/v1/object/public/chat-images/${storagePath}` : undefined)

  return (
    <div
      className={cn(
        'relative inline-flex items-center gap-2.5 p-1.5 pr-2.5 border rounded-xl bg-card/80 shadow-2xs transition-all duration-150',
        attachment.error ? 'border-destructive/40 bg-destructive/5' : 'border-border/60 hover:border-border',
      )}
    >
      {/* Thumbnail or File Icon */}
      {isImage && imageUrl ? (
        <div className="relative h-10 w-10 rounded-lg overflow-hidden shrink-0 border border-border/40 bg-background">
          <img
            src={imageUrl}
            alt={filename}
            className="h-full w-full object-cover"
          />
        </div>
      ) : (
        <div className="flex items-center justify-center h-10 w-10 rounded-lg shrink-0 border border-border/30 bg-muted/50">
          {getFileIcon(filename, mimeType)}
        </div>
      )}

      {/* File Details & Status */}
      <div className="flex flex-col min-w-0 pr-1 max-w-48">
        <span className="text-xs font-medium text-foreground truncate" title={filename}>
          {filename}
        </span>
        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          {attachment.size > 0 && <span>{formatSize(attachment.size)}</span>}
          {attachment.status === 'uploading' && (
            <span className="text-primary font-medium">Uploading {attachment.progress ? `${attachment.progress}%` : ''}</span>
          )}
          {attachment.error && (
            <span className="text-destructive flex items-center gap-0.5 truncate">
              <AlertCircle className="h-3 w-3 shrink-0" />
              {attachment.error}
            </span>
          )}
        </div>

        {/* Upload Progress Bar */}
        {attachment.status === 'uploading' && typeof attachment.progress === 'number' && (
          <div className="w-full bg-muted rounded-full h-1 mt-1 overflow-hidden">
            <div
              className="bg-primary h-full transition-all duration-200"
              style={{ width: `${attachment.progress}%` }}
            />
          </div>
        )}
      </div>

      {/* Remove Button */}
      {!readOnly && onRemove && (
        <button
          type="button"
          onClick={() => onRemove(attachment.id)}
          className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted rounded-full transition-colors ml-auto"
          title="Remove attachment"
          aria-label="Remove attachment"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  )
}
