import { UploadCloudIcon, FileUpIcon } from 'lucide-react'
import { useRef, useState, type DragEvent, type KeyboardEvent } from 'react'

import { Button } from '@/components/ui/button'

interface UploadDropzoneProps {
  onFilesSelected: (files: FileList | File[]) => void
  disabled?: boolean
}

export function UploadDropzone({ onFilesSelected, disabled = false }: UploadDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (!disabled) {
      setIsDragOver(true)
    }
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)

    if (disabled) return

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(e.dataTransfer.files)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files)
      // Reset input value so re-selecting the same file fires change event
      e.target.value = ''
    }
  }

  const triggerFileInput = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
      e.preventDefault()
      triggerFileInput()
    }
  }

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={triggerFileInput}
      onKeyDown={handleKeyDown}
      className={`
        relative group rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer p-8 md:p-12
        flex flex-col items-center justify-center text-center
        ${
          isDragOver
            ? 'border-primary bg-primary/5 shadow-lg scale-[1.005]'
            : 'border-border/80 hover:border-primary/60 bg-card hover:bg-accent/40 shadow-xs'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed pointer-events-none' : ''}
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.txt,.md,.markdown"
        onChange={handleFileChange}
        className="sr-only"
        aria-label="Upload document files"
      />

      <div
        className={`
        p-4 rounded-full transition-all duration-200 mb-4
        ${
          isDragOver
            ? 'bg-primary/20 text-primary scale-110'
            : 'bg-muted/80 text-muted-foreground group-hover:bg-primary/10 group-hover:text-primary'
        }
      `}
      >
        {isDragOver ? (
          <FileUpIcon className="size-10 animate-bounce" />
        ) : (
          <UploadCloudIcon className="size-10" />
        )}
      </div>

      <div className="space-y-1 max-w-md">
        <h3 className="text-base font-semibold text-foreground">
          {isDragOver ? 'Drop files here to queue' : 'Drag & drop files here'}
        </h3>
        <p className="text-sm text-muted-foreground">
          or{' '}
          <span className="font-medium text-primary underline underline-offset-2">
            browse files
          </span>{' '}
          from your computer
        </p>
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-xs text-muted-foreground/80 border-t border-border/40 pt-4 w-full max-w-md">
        <span>Accepted: <strong className="font-mono text-foreground/80">.pdf, .docx, .txt, .md</strong></span>
        <span>•</span>
        <span>Max size: <strong className="font-mono text-foreground/80">25 MB</strong></span>
      </div>

      <Button
        type="button"
        variant="secondary"
        size="sm"
        disabled={disabled}
        onClick={(e) => {
          e.stopPropagation()
          triggerFileInput()
        }}
        className="mt-4 shadow-2xs pointer-events-auto"
      >
        Select Files
      </Button>
    </div>
  )
}
