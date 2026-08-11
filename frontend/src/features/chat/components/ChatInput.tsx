import { useState, useRef, useEffect } from 'react'
import { SendHorizontal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSend: (message: string) => void
  /** Disables the entire input (rare). Prefer sendDisabled while generating. */
  disabled?: boolean
  /** Prevents duplicate Send for the active conversation only. Input stays visible/usable. */
  sendDisabled?: boolean
  placeholder?: string
}

export function ChatInput({
  onSend,
  disabled,
  sendDisabled,
  placeholder = 'Ask a question about your documents...',
}: ChatInputProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const cannotSend = Boolean(disabled || sendDisabled)

  const handleInput = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }

  useEffect(() => {
    handleInput()
  }, [input])

  const handleSend = () => {
    console.log("[1] ChatInput.handleSend()");
    console.log("Question:", input);
    const trimmed = input.trim()
    if (!trimmed || cannotSend) return
    onSend(trimmed)
    setInput('')
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

  return (
    <div className="relative flex items-end w-full rounded-xl border border-input bg-background p-2 shadow-sm focus-within:ring-1 focus-within:ring-ring">
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
            : placeholder
        }
        disabled={disabled}
        className={cn(
          'flex-1 min-h-10 max-h-50 w-full resize-none bg-transparent px-3 py-2 text-sm focus:outline-none disabled:cursor-not-allowed disabled:opacity-50',
          'scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent',
        )}
        rows={1}
      />
      <Button
        size="icon"
        onClick={handleSend}
        disabled={!input.trim() || cannotSend}
        className="ml-2 mb-1 h-8 w-8 shrink-0 rounded-lg"
        title={sendDisabled ? 'Wait for the current response in this chat' : 'Send message'}
      >
        <SendHorizontal className="h-4 w-4" />
        <span className="sr-only">Send message</span>
      </Button>
    </div>
  )
}
