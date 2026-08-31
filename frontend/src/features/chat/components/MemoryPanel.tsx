import { Brain, Trash2, Plus, Sparkles, RefreshCw } from 'lucide-react'
import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { memoryService, type LongTermMemory } from '../services/memory.service'

interface MemoryPanelProps {
  isOpen: boolean
  onToggle: () => void
}

const TYPE_COLORS: Record<string, string> = {
  preference: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20',
  goal: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20',
  technical_context: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20',
  user_profile: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20',
  project_context: 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/20',
  decision: 'bg-teal-500/10 text-teal-600 dark:text-teal-400 border-teal-500/20',
  requirement: 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20',
  other: 'bg-slate-500/10 text-slate-600 dark:text-slate-400 border-slate-500/20',
}

export function MemoryPanel({ isOpen, onToggle }: MemoryPanelProps) {
  const [memories, setMemories] = useState<LongTermMemory[]>([])
  const [loading, setLoading] = useState(false)
  const [newContent, setNewContent] = useState('')
  const [adding, setAdding] = useState(false)

  const fetchMemories = async () => {
    setLoading(true)
    try {
      const data = await memoryService.listMemories({ is_active: true })
      setMemories(data.items || [])
    } catch (err) {
      console.warn('Failed to load memories:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      fetchMemories()
    }
  }, [isOpen])

  const handleDelete = async (id: string) => {
    try {
      await memoryService.deleteMemory(id)
      setMemories(prev => prev.filter(m => m.id !== id))
    } catch (err) {
      console.error('Failed to delete memory:', err)
    }
  }

  const handlePurgeAll = async () => {
    if (!window.confirm('Are you sure you want to clear all learned long-term memories?')) return
    try {
      await memoryService.purgeAllMemories()
      setMemories([])
    } catch (err) {
      console.error('Failed to purge memories:', err)
    }
  }

  const handleAddManual = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newContent.trim()) return
    setAdding(true)
    try {
      const created = await memoryService.createMemory({
        content: newContent.trim(),
        memory_type: 'preference',
        importance: 0.8,
      })
      setMemories(prev => [created, ...prev])
      setNewContent('')
    } catch (err) {
      console.error('Failed to create memory:', err)
    } finally {
      setAdding(false)
    }
  }

  return (
    <div className="border-t border-border/30 bg-muted/5">
      <button
        onClick={onToggle}
        className="w-full px-4 py-2.5 flex items-center justify-between text-xs font-semibold text-muted-foreground hover:text-foreground hover:bg-muted/10 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Brain className="w-3.5 h-3.5 text-primary" />
          <span>Learned Memories</span>
          {memories.length > 0 && (
            <span className="px-1.5 py-0.5 rounded-full text-[10px] bg-primary/10 text-primary font-bold">
              {memories.length}
            </span>
          )}
        </div>
        <span className="text-[10px] text-muted-foreground/60">
          {isOpen ? 'Hide' : 'Show'}
        </span>
      </button>

      {isOpen && (
        <div className="p-3 flex flex-col gap-2 max-h-64 overflow-y-auto border-t border-border/20 text-xs">
          <div className="flex items-center justify-between text-[11px] text-muted-foreground mb-1">
            <span className="flex items-center gap-1 font-medium">
              <Sparkles className="w-3 h-3 text-amber-500" />
              Auto-extracted & manual facts
            </span>
            <div className="flex items-center gap-1">
              <button
                onClick={fetchMemories}
                disabled={loading}
                title="Refresh memories"
                className="p-1 hover:bg-muted/20 rounded transition-colors text-muted-foreground hover:text-foreground"
              >
                <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
              </button>
              {memories.length > 0 && (
                <button
                  onClick={handlePurgeAll}
                  title="Clear all memories"
                  className="p-1 hover:bg-destructive/10 rounded transition-colors text-destructive/70 hover:text-destructive text-[10px] font-medium"
                >
                  Clear All
                </button>
              )}
            </div>
          </div>

          <form onSubmit={handleAddManual} className="flex items-center gap-1.5 mb-1">
            <Input
              placeholder="Add a preference or fact to remember..."
              value={newContent}
              onChange={e => setNewContent(e.target.value)}
              className="h-7 text-[11px] bg-background/50 border-border/30 rounded"
              disabled={adding}
            />
            <Button
              type="submit"
              size="sm"
              variant="outline"
              className="h-7 px-2 text-[11px] shrink-0"
              disabled={adding || !newContent.trim()}
            >
              <Plus className="w-3 h-3" />
            </Button>
          </form>

          {loading && memories.length === 0 ? (
            <div className="py-4 text-center text-muted-foreground/60 text-[11px]">
              Loading memories...
            </div>
          ) : memories.length === 0 ? (
            <div className="py-3 text-center text-muted-foreground/50 text-[11px] italic">
              No memories saved yet. State preferences in chat to auto-save!
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {memories.map(mem => {
                const badgeStyle = TYPE_COLORS[mem.memory_type] || TYPE_COLORS.other
                return (
                  <div
                    key={mem.id}
                    className="p-2 rounded-md bg-background/60 border border-border/30 flex items-start justify-between gap-2 group hover:border-border/60 transition-colors"
                  >
                    <div className="flex flex-col gap-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <Badge
                          variant="outline"
                          className={`text-[9px] px-1.5 py-0 capitalize font-medium ${badgeStyle}`}
                        >
                          {mem.memory_type.replace('_', ' ')}
                        </Badge>
                      </div>
                      <p className="text-[11px] leading-snug text-foreground/90 wrap-break-word">
                        {mem.content}
                      </p>
                    </div>
                    <button
                      onClick={() => handleDelete(mem.id)}
                      title="Delete memory"
                      className="opacity-0 group-hover:opacity-100 p-1 text-muted-foreground/60 hover:text-destructive hover:bg-destructive/10 rounded transition-all shrink-0"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
