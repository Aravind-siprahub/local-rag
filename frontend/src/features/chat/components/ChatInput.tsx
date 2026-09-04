import { useState, useRef, useEffect, useCallback } from 'react'
import { SendHorizontal, Paperclip, Pencil, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { ModelSelector } from './ModelSelector'
import { AttachmentCard } from './AttachmentCard'
import { DragDropOverlay } from './DragDropOverlay'
import type { Message, Attachment } from '../types/chat'

import { uploadDocument } from '@/services/upload.service'
import { useAuth } from '@/features/auth/hooks/authHooks'

const SUPPORTED_EXTENSIONS = [
  'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls', 'pptx', 'ppt',
  'png', 'jpg', 'jpeg', 'webp'
]

const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp']

interface ChatInputProps {
  onSend: (message: string, attachments?: Attachment[], primaryFile?: File, preservedImageUrl?: string) => void
  disabled?: boolean
  sendDisabled?: boolean
  placeholder?: string
  selectedModel: string
  onSelectModel: (modelId: string) => void
  editingMessage?: Message | null
  onCancelEdit?: () => void
  onStop?: () => void
}

export function ChatInput({
  onSend,
  disabled = false,
  sendDisabled = false,
  placeholder = 'Ask any question...',
  selectedModel,
  onSelectModel,
  editingMessage,
  onCancelEdit,
  onStop,
}: ChatInputProps) {
  const { user } = useAuth()
  const isAdmin = user?.role?.toLowerCase() === 'admin'

  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [preservedImageUrl, setPreservedImageUrl] = useState<string | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const dragCounterRef = useRef(0)
  const cannotSend = Boolean(disabled || sendDisabled)

  // Sync state when entering Edit mode
  useEffect(() => {
    if (editingMessage) {
      setInput(editingMessage.content || '')
      if (editingMessage.attachments && editingMessage.attachments.length > 0) {
        setAttachments(editingMessage.attachments)
      }
      if (editingMessage.localImageUrl) {
        setPreservedImageUrl(editingMessage.localImageUrl)
      } else {
        setPreservedImageUrl(null)
      }
      setErrorMsg(null)
      if (textareaRef.current) {
        textareaRef.current.focus()
      }
    }
  }, [editingMessage])

  const handleInput = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`
    }
  }

  useEffect(() => {
    handleInput()
  }, [input])

  const processFiles = (files: FileList | File[]) => {
    setErrorMsg(null)
    const newAttachments: Attachment[] = []

    Array.from(files).forEach((file) => {
      const ext = file.name.split('.').pop()?.toLowerCase() || ''

      if (!SUPPORTED_EXTENSIONS.includes(ext)) {
        setErrorMsg(`Unsupported file type: .${ext}`)
        return
      }

      const isImg = IMAGE_EXTENSIONS.includes(ext)
      if (!isImg && !isAdmin) {
        setErrorMsg('Only administrators have permission to upload documents.')
        return
      }

      const maxSize = isImg ? 10 * 1024 * 1024 : 25 * 1024 * 1024

      if (file.size > maxSize) {
        setErrorMsg(`${file.name} exceeds maximum allowed size (${isImg ? '10 MB' : '25 MB'}).`)
        return
      }

      const attId = `att-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
      const previewUrl = isImg ? URL.createObjectURL(file) : undefined

      const att: Attachment = {
        id: attId,
        filename: file.name,
        mime_type: file.type || `application/${ext}`,
        size: file.size,
        file: file,
        previewUrl: previewUrl,
        status: isImg ? 'ready' : 'uploading',
        progress: isImg ? 100 : 10,
      }

      newAttachments.push(att)

      // Automatically upload documents so backend receives document_id & parses chunks
      if (!isImg) {
        const userId = localStorage.getItem('USER_ID') || 'demo-user'
        uploadDocument({
          file,
          userId,
          title: file.name,
          onUploadProgress: (pct) => {
            setAttachments((prev) =>
              prev.map((a) => (a.id === attId ? { ...a, progress: pct } : a))
            )
          },
        })
          .then((res) => {
            setAttachments((prev) =>
              prev.map((a) =>
                a.id === attId
                  ? {
                      ...a,
                      status: 'ready',
                      progress: 100,
                      document_id: res.document_id || (res as unknown as { id?: string }).id,
                    }
                  : a
              )
            )
          })
          .catch((err) => {
            console.error('[DOCUMENT UPLOAD FAILED]', err)
            setAttachments((prev) =>
              prev.map((a) =>
                a.id === attId
                  ? { ...a, status: 'error', error: 'Document processing failed' }
                  : a
              )
            )
          })
      }
    })

    if (newAttachments.length > 0) {
      setAttachments((prev) => [...prev, ...newAttachments])
      setPreservedImageUrl(null)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files)
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const removeAttachment = (id: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.id === id)
      if (target?.previewUrl) {
        URL.revokeObjectURL(target.previewUrl)
      }
      return prev.filter((a) => a.id !== id)
    })
  }

  const clearAllAttachments = (revoke = false) => {
    if (revoke) {
      attachments.forEach((a) => {
        if (a.previewUrl) URL.revokeObjectURL(a.previewUrl)
      })
    }
    setAttachments([])
    setPreservedImageUrl(null)
    setErrorMsg(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Drag and Drop Handlers
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounterRef.current += 1
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true)
    }
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounterRef.current -= 1
    if (dragCounterRef.current === 0) {
      setIsDragging(false)
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    dragCounterRef.current = 0

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files)
      e.dataTransfer.clearData()
    }
  }, [])

  const handleSend = () => {
    const trimmed = input.trim()
    const hasAttachments = attachments.length > 0 || Boolean(preservedImageUrl)

    if ((!trimmed && !hasAttachments) || cannotSend) return

    // Pick primary file if image attached for backward compatibility
    const firstImageAtt = attachments.find((a) =>
      a.mime_type.startsWith('image/') || IMAGE_EXTENSIONS.some((ext) => a.filename.toLowerCase().endsWith(`.${ext}`))
    )
    const primaryFile = firstImageAtt?.file

    onSend(trimmed, attachments, primaryFile, preservedImageUrl || undefined)
    setInput('')
    clearAllAttachments(false)
    if (onCancelEdit) onCancelEdit()
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    } else if (e.key === 'Escape' && editingMessage) {
      e.preventDefault()
      setInput('')
      clearAllAttachments()
      if (onCancelEdit) onCancelEdit()
    }
  }

  const hasActiveImage = attachments.some((a) =>
    a.mime_type.startsWith('image/') || IMAGE_EXTENSIONS.some((ext) => a.filename.toLowerCase().endsWith(`.${ext}`))
  ) || Boolean(preservedImageUrl)

  return (
    <div
      className="relative flex flex-col w-full gap-2"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <DragDropOverlay isDragging={isDragging} />

      {/* Editing Message Banner */}
      {editingMessage && (
        <div className="flex items-center justify-between px-3 py-1.5 bg-primary/10 border border-primary/20 rounded-xl text-xs text-primary animate-in fade-in-0 duration-150">
          <div className="flex items-center gap-1.5 font-medium truncate">
            <Pencil className="h-3.5 w-3.5 shrink-0" />
            <span>Editing user message</span>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => {
              setInput('')
              clearAllAttachments()
              if (onCancelEdit) onCancelEdit()
            }}
            className="h-6 px-2 text-[11px] text-muted-foreground hover:text-foreground hover:bg-background/60 rounded-lg"
          >
            Cancel edit
          </Button>
        </div>
      )}

      {errorMsg && (
        <div className="text-xs text-destructive px-3 py-1.5 bg-destructive/10 rounded-lg border border-destructive/20 max-w-fit self-start animate-in fade-in-0">
          {errorMsg}
        </div>
      )}

      {editingMessage && preservedImageUrl && attachments.length === 0 && (
        <div className="text-[11px] text-amber-500 px-3 py-1 bg-amber-500/10 rounded-lg border border-amber-500/20 max-w-fit self-start animate-in fade-in-0">
          <strong>Note:</strong> To include the original image in your resubmission, you must re-attach it.
        </div>
      )}

      <div
        className={cn(
          'relative flex flex-col w-full rounded-xl border border-border/40 bg-card/50 p-2 shadow-xs transition-all duration-200 focus-within:border-primary/40 focus-within:ring-1 focus-within:ring-primary/30',
          cannotSend && 'opacity-80',
        )}
      >
        {/* Attachment Previews Bar inside Composer */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 p-1.5 mb-1 max-h-36 overflow-y-auto scrollbar-thin scrollbar-thumb-muted-foreground/20">
            {attachments.map((att) => (
              <AttachmentCard key={att.id} attachment={att} onRemove={removeAttachment} />
            ))}
          </div>
        )}

        <div className="relative flex flex-col w-full">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            multiple
            accept=".pdf,.docx,.doc,.txt,.csv,.xlsx,.xls,.pptx,.ppt,.png,.jpg,.jpeg,.webp"
            className="hidden"
            id="chat-file-input"
          />

          <textarea
            id="chat-input"
            name="message"
            autoComplete="off"
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              sendDisabled && !disabled
                ? 'Generating response… you can still browse other chats'
                : editingMessage
                ? 'Modify your message and press Resubmit...'
                : placeholder
            }
            disabled={disabled}
            className={cn(
              'flex-1 min-h-11 max-h-56 w-full resize-none bg-transparent px-3 py-3 text-sm focus:outline-none disabled:cursor-not-allowed disabled:opacity-50 leading-relaxed text-foreground placeholder:text-muted-foreground/50',
              'scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent',
            )}
            rows={1}
          />

          <div className="flex items-center justify-between pt-1 px-1">
            <div className="flex items-center gap-1.5">
              {isAdmin && (
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={disabled}
                  className={cn(
                    'h-8 w-8 rounded-lg text-muted-foreground hover:text-foreground transition-colors',
                    attachments.length > 0 && 'text-primary bg-primary/10 hover:bg-primary/20',
                  )}
                  title="Attach files (PDF, DOCX, CSV, XLSX, Images...)"
                  aria-label="Attach files"
                >
                  <Paperclip className="h-4 w-4" />
                </Button>
              )}

              <ModelSelector
                selectedModel={selectedModel}
                onSelectModel={onSelectModel}
                hasImageAttached={hasActiveImage}
                disabled={cannotSend}
              />
            </div>

            <Button
              size="icon"
              onClick={sendDisabled && onStop ? onStop : handleSend}
              disabled={sendDisabled ? false : ((!input.trim() && attachments.length === 0 && !preservedImageUrl) || cannotSend)}
              className={cn(
                'h-8 w-8 shrink-0 rounded-lg shadow-xs transition-all active:scale-95 border border-transparent',
                sendDisabled
                  ? 'bg-destructive/10 text-destructive hover:bg-destructive/20 border-destructive/20'
                  : 'bg-primary hover:bg-primary/90 text-primary-foreground disabled:bg-muted/40 disabled:text-muted-foreground/30 disabled:border-border/10'
              )}
              title={
                sendDisabled
                  ? 'Stop generating'
                  : editingMessage
                  ? 'Resubmit edited message'
                  : 'Send message'
              }
              aria-label={sendDisabled ? 'Stop generating' : editingMessage ? 'Resubmit edited message' : 'Send message'}
            >
              {sendDisabled ? <Square className="h-3 w-3 fill-current" /> : <SendHorizontal className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
