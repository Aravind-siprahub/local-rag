import { useState, useRef, useEffect } from 'react'
import { ChevronDown, Check, Image as ImageIcon, Cpu } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ModelOption {
  id: string
  name: string
  description: string
  type: 'text' | 'vision'
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'qwen3:4b',
    name: 'qwen3:4b',
    description: 'Text and Document RAG',
    type: 'text',
  },
  {
    id: 'qwen3-vl:4b',
    name: 'qwen3-vl:4b',
    description: 'Multimodal Image Analysis',
    type: 'vision',
  },
]

interface ModelSelectorProps {
  selectedModel: string
  onSelectModel: (modelId: string) => void
  hasImageAttached?: boolean
  disabled?: boolean
  placement?: 'top' | 'bottom'
}

export function ModelSelector({
  selectedModel,
  onSelectModel,
  hasImageAttached,
  disabled,
  placement = 'top',
}: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-switch / lock to vision model if image is attached
  const activeModelId = hasImageAttached ? 'qwen3-vl:4b' : selectedModel
  const currentModel =
    AVAILABLE_MODELS.find((m) => m.id === activeModelId) || AVAILABLE_MODELS[0]

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  return (
    <div className="relative inline-block text-left" ref={containerRef}>
      <button
        type="button"
        onClick={() => !disabled && setIsOpen((prev) => !prev)}
        disabled={disabled}
        className={cn(
          'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono border transition-all duration-150 shadow-xs focus:outline-none focus:ring-1 focus:ring-primary/40',
          hasImageAttached
            ? 'bg-primary/10 border-primary/30 text-primary font-medium'
            : 'bg-muted/40 border-border/60 text-muted-foreground hover:text-foreground hover:bg-muted/70',
          disabled && 'opacity-50 cursor-not-allowed',
        )}
        title={
          hasImageAttached
            ? 'Automatically using qwen3-vl:4b for image analysis'
            : 'Select AI Model'
        }
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        {hasImageAttached ? (
          <ImageIcon className="h-3.5 w-3.5 text-primary shrink-0 animate-pulse" />
        ) : (
          <Cpu className="h-3.5 w-3.5 shrink-0" />
        )}
        <span className="truncate">{currentModel.name}</span>
        <ChevronDown
          className={cn(
            'h-3 w-3 transition-transform duration-150',
            isOpen && (placement === 'top' ? 'rotate-180' : 'rotate-180'),
          )}
        />
      </button>

      {isOpen && (
        <div
          className={cn(
            'absolute right-0 w-64 rounded-xl border border-border/70 bg-card p-1.5 shadow-xl z-50 animate-in fade-in-0 zoom-in-95',
            placement === 'top' ? 'bottom-full mb-2' : 'top-full mt-1.5',
          )}
        >
          <div className="px-2 py-1 mb-1 border-b border-border/40 flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Select Model
            </span>
            {hasImageAttached && (
              <span className="text-[10px] bg-primary/15 text-primary px-1.5 py-0.5 rounded font-medium">
                Auto Vision
              </span>
            )}
          </div>

          <div className="space-y-0.5" role="listbox">
            {AVAILABLE_MODELS.map((model) => {
              const isSelected = model.id === currentModel.id
              const isAutoVisionLocked = hasImageAttached && model.id !== 'qwen3-vl:4b'

              return (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => {
                    if (!isAutoVisionLocked) {
                      onSelectModel(model.id)
                      setIsOpen(false)
                    }
                  }}
                  disabled={isAutoVisionLocked}
                  className={cn(
                    'w-full flex items-start gap-2.5 p-2 rounded-lg text-left transition-colors text-xs',
                    isSelected ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted/60 text-foreground',
                    isAutoVisionLocked && 'opacity-40 cursor-not-allowed hover:bg-transparent',
                  )}
                  role="option"
                  aria-selected={isSelected}
                >
                  <div className="mt-0.5 shrink-0">
                    {isSelected ? (
                      <Check className="h-3.5 w-3.5 text-primary" />
                    ) : (
                      <div className="h-3.5 w-3.5" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-xs font-semibold">{model.name}</span>
                      {model.type === 'vision' && (
                        <span className="text-[9px] bg-accent text-accent-foreground px-1 rounded font-sans">
                          Vision
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-tight mt-0.5">
                      {model.description}
                    </p>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
