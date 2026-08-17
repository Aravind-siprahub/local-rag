import { useState, useRef, useEffect } from 'react'
import { SendHorizontal, Paperclip, X, Image as ImageIcon, Pencil } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { ModelSelector } from './ModelSelector'
import type { Message } from '../types/chat'

interface ChatInputProps {
  onSend: (message: string, file?: File, preservedImageUrl?: string) => void
  disabled?: boolean
  sendDisabled?: boolean
  placeholder?: string
  selectedModel: string
  onSelectModel: (modelId: string) => void
  editingMessage?: Message | null
  onCancelEdit?: () => void
}

export function ChatInput({
  onSend,
  disabled,
  sendDisabled,
  placeholder = 'Ask a question about your documents...',
  selectedModel,
  onSelectModel,
  editingMessage,
  onCancelEdit,
}: ChatInputProps) {
  const [input, setInput] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [preservedImageUrl, setPreservedImageUrl] = useState<string | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const cannotSend = Boolean(disabled || sendDisabled)

  // Sync state when entering Edit mode
  useEffect(() => {
    if (editingMessage) {
      setInput(editingMessage.content || '')
      if (editingMessage.localImageUrl) {
        setPreservedImageUrl(editingMessage.localImageUrl)
        setPreviewUrl(editingMessage.localImageUrl)
      } else {
        setPreservedImageUrl(null)
        setPreviewUrl(null)
      }
      setSelectedFile(null)
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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMsg(null)
    const file = e.target.files?.[0]
    if (!file) return

    // Size limit: 10 MB
    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg('Image size exceeds the 10 MB limit.')
      return
    }

    // MIME type check
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp']
    if (!validTypes.includes(file.type)) {
      setErrorMsg('Unsupported image format. Only PNG, JPEG, and WEBP are supported.')
      return
    }

    setSelectedFile(file)
    setPreservedImageUrl(null) // New file replaces preserved image
    const objectUrl = URL.createObjectURL(file)
    setPreviewUrl(objectUrl)
  }

  const clearFile = () => {
    setSelectedFile(null)
    if (previewUrl && !preservedImageUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setPreviewUrl(null)
    setPreservedImageUrl(null)
    setErrorMsg(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleSend = () => {
    const trimmed = input.trim()
    if ((!trimmed && !selectedFile && !preservedImageUrl) || cannotSend) return
    onSend(trimmed, selectedFile || undefined, preservedImageUrl || undefined)
    setInput('')
    clearFile()
    if (onCancelEdit) {
      onCancelEdit()
    }
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const hasActiveImage = Boolean(previewUrl || preservedImageUrl || selectedFile)

  return (
    <div className="flex flex-col w-full gap-2">
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
              clearFile()
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

      {editingMessage && preservedImageUrl && !selectedFile && (
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
        {/* Compact Image Preview Bar inside Composer */}
        {previewUrl && (
          <div className="relative inline-flex items-center gap-3 m-1.5 p-1.5 pr-3 border border-border/60 rounded-xl bg-muted/40 self-start animate-in fade-in-0 duration-150">
            <div className="relative h-12 w-12 rounded-lg overflow-hidden shrink-0 border border-border/40 bg-background">
              <img
                src={previewUrl}
                alt="Attached preview"
                className="h-full w-full object-cover"
              />
            </div>
            <div className="flex flex-col min-w-0 pr-1">
              <span className="text-xs font-medium text-foreground truncate max-w-44">
                {selectedFile?.name || 'Original Image'}
              </span>
              <span className="text-[10px] text-muted-foreground">
                {selectedFile ? (
                  `${(selectedFile.size / 1024).toFixed(1)} KB`
                ) : (
                  <span className="text-amber-500 font-medium">Re-attach required</span>
                )}
              </span>
            </div>
            <button
              type="button"
              onClick={clearFile}
              className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted rounded-full transition-colors ml-auto"
              title="Remove attached image"
              aria-label="Remove attached image"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <div className="relative flex items-end w-full">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/png, image/jpeg, image/jpg, image/webp"
            className="hidden"
            id="chat-image-input"
          />

          <Button
            type="button"
            size="icon"
            variant="ghost"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className={cn(
              'h-8 w-8 mr-1 mb-0.5 shrink-0 rounded-lg text-muted-foreground/60 hover:text-foreground transition-colors',
              hasActiveImage && 'text-primary bg-primary/10 hover:bg-primary/20',
            )}
            title="Attach image (PNG, JPEG, WEBP)"
            aria-label="Attach image"
          >
            {hasActiveImage ? <ImageIcon className="h-3.5 w-3.5" /> : <Paperclip className="h-3.5 w-3.5" />}
          </Button>

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
              'flex-1 min-h-8 max-h-44 w-full resize-none bg-transparent px-2 py-1.5 text-xs focus:outline-none disabled:cursor-not-allowed disabled:opacity-50 leading-relaxed text-foreground placeholder:text-muted-foreground/40',
              'scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent',
            )}
            rows={1}
          />

          {/* Embedded Model Selector */}
          <div className="mb-1 mr-1 shrink-0">
            <ModelSelector
              selectedModel={selectedModel}
              onSelectModel={onSelectModel}
              hasImageAttached={hasActiveImage}
              disabled={cannotSend}
            />
          </div>

          <Button
            size="icon"
            onClick={handleSend}
            disabled={(!input.trim() && !hasActiveImage) || cannotSend}
            className="ml-1 mb-0.5 h-8 w-8 shrink-0 rounded-lg shadow-xs transition-all active:scale-95 bg-primary hover:bg-primary/90 text-primary-foreground disabled:bg-muted/40 disabled:text-muted-foreground/30 border border-transparent disabled:border-border/10"
            title={
              sendDisabled
                ? 'Wait for the current response'
                : editingMessage
                ? 'Resubmit edited message'
                : 'Send message'
            }
            aria-label={editingMessage ? 'Resubmit edited message' : 'Send message'}
          >
            <SendHorizontal className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    </div>
  )
}
