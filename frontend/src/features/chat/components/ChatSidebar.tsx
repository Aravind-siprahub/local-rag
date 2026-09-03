import { Plus, MessageSquare, Trash2, Search, X, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'
import { formatDistanceToNow, isToday, isYesterday, subDays, isAfter } from 'date-fns'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import type { Conversation } from '../types/chat'
import { MemoryPanel } from './MemoryPanel'

interface ChatSidebarProps {
  conversations: Conversation[]
  activeId?: string
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  isMobileOpen: boolean
  setMobileOpen: (open: boolean) => void
}

interface GroupedConversations {
  title: string
  items: Conversation[]
}

function groupConversationsByDate(conversations: Conversation[]): GroupedConversations[] {
  const today: Conversation[] = []
  const yesterday: Conversation[] = []
  const previous7Days: Conversation[] = []
  const previous30Days: Conversation[] = []
  const older: Conversation[] = []

  const now = new Date()
  const sevenDaysAgo = subDays(now, 7)
  const thirtyDaysAgo = subDays(now, 30)

  for (const conv of conversations) {
    const date = conv.last_message_at ? new Date(conv.last_message_at) : new Date(conv.created_at || now)
    if (isToday(date)) {
      today.push(conv)
    } else if (isYesterday(date)) {
      yesterday.push(conv)
    } else if (isAfter(date, sevenDaysAgo)) {
      previous7Days.push(conv)
    } else if (isAfter(date, thirtyDaysAgo)) {
      previous30Days.push(conv)
    } else {
      older.push(conv)
    }
  }

  const groups: GroupedConversations[] = []
  if (today.length > 0) groups.push({ title: 'Today', items: today })
  if (yesterday.length > 0) groups.push({ title: 'Yesterday', items: yesterday })
  if (previous7Days.length > 0) groups.push({ title: 'Previous 7 Days', items: previous7Days })
  if (previous30Days.length > 0) groups.push({ title: 'Previous 30 Days', items: previous30Days })
  if (older.length > 0) groups.push({ title: 'Older', items: older })

  return groups
}

const INITIAL_LIMIT = 3

export function ChatSidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  isMobileOpen,
  setMobileOpen,
}: ChatSidebarProps) {
  const [search, setSearch] = useState('')
  const [showAll, setShowAll] = useState(false)
  const [isMemoryOpen, setIsMemoryOpen] = useState(false)


  const filtered = conversations.filter(c => 
    c.title.toLowerCase().includes(search.toLowerCase())
  )

  const displayedConversations = showAll || search.trim().length > 0
    ? filtered
    : filtered.slice(0, INITIAL_LIMIT)

  const groups = groupConversationsByDate(displayedConversations)

  const SidebarContent = (
    <div className="flex flex-col h-full w-full bg-muted/5 border-r border-border/30 overflow-hidden">
      <div className="p-4 flex flex-col gap-3 shrink-0">
        <Button 
          onClick={onNew} 
          className="w-full justify-start h-9 text-xs font-semibold rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground shadow-xs active:scale-[0.98] transition-transform"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Chat
        </Button>
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground/40" />
          <Input
            placeholder="Search conversations..."
            className="pl-8.5 h-8 bg-muted/10 border-border/20 rounded-lg text-xs placeholder:text-muted-foreground/40 focus-visible:ring-primary/20 focus-visible:border-primary/40 transition-colors"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
      </div>
      
      <div className="h-px bg-border/20 mx-4 mb-2 shrink-0" />
      
      <div className="flex-1 overflow-y-auto px-2 py-1 flex flex-col gap-1 min-h-0 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40">
          {filtered.length === 0 ? (
            <p className="text-[11px] text-muted-foreground/50 text-center p-4">
              No conversations found.
            </p>
          ) : (
            <>
              {groups.map((group) => (
                <div key={group.title} className="flex flex-col gap-0.5 mb-1">
                  <div className="text-[10px] font-semibold text-muted-foreground/60 uppercase tracking-wider px-2.5 py-1 select-none">
                    {group.title}
                  </div>
                  {group.items.map((conv) => (
                    <div
                      key={conv.id}
                      className={cn(
                        "group flex items-center justify-between px-2.5 py-2 rounded-lg cursor-pointer transition-all duration-150 relative",
                        activeId === conv.id 
                          ? "bg-muted/40 text-foreground font-medium border border-border/20" 
                          : "hover:bg-muted/20 text-muted-foreground hover:text-foreground border border-transparent"
                      )}
                      onClick={() => {
                        onSelect(conv.id)
                        setMobileOpen(false)
                      }}
                    >
                      <div className="flex items-center min-w-0 overflow-hidden pr-6">
                        <MessageSquare className="w-3.5 h-3.5 mr-2.5 shrink-0 opacity-40 group-hover:opacity-75 transition-opacity" />
                        <div className="flex flex-col truncate">
                          <span className="text-xs truncate max-w-38.75 text-foreground/90">{conv.title}</span>
                          {conv.last_message_at && (
                            <span className="text-[9px] text-muted-foreground/50 mt-0.5">
                              {formatDistanceToNow(new Date(conv.last_message_at), { addSuffix: true })}
                            </span>
                          )}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className={cn(
                          "h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md hover:bg-destructive/10 hover:text-destructive text-muted-foreground/40",
                          activeId === conv.id ? "opacity-100" : ""
                        )}
                        onClick={(e) => {
                          e.preventDefault()
                          e.stopPropagation()
                          onDelete(conv.id)
                        }}
                        title="Delete chat"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  ))}
                </div>
              ))}

              {filtered.length > INITIAL_LIMIT && !search.trim() && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowAll(prev => !prev)}
                  className="w-full text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30 h-8 justify-center gap-1.5 mt-1 rounded-lg transition-colors font-medium"
                >
                  {showAll ? (
                    <>
                      <ChevronUp className="w-3.5 h-3.5" />
                      <span>Show Less</span>
                    </>
                  ) : (
                    <>
                      <ChevronDown className="w-3.5 h-3.5" />
                      <span>Show More ({filtered.length - INITIAL_LIMIT} more)</span>
                    </>
                  )}
                </Button>
              )}
            </>
          )}
      </div>

      <MemoryPanel isOpen={isMemoryOpen} onToggle={() => setIsMemoryOpen(prev => !prev)} />
    </div>
  )

  return (
    <>
      {/* Desktop Sidebar */}
      <div className="hidden md:flex w-72 flex-col h-full shrink-0">
        {SidebarContent}
      </div>

      {/* Mobile Sidebar Overlay */}
      {isMobileOpen && (
        <div 
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}
      
      {/* Mobile Sidebar */}
      <div className={cn(
        "fixed inset-y-0 left-0 z-50 w-72 bg-background transform transition-transform duration-200 ease-in-out md:hidden flex flex-col",
        isMobileOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="font-semibold">Conversations</h2>
          <Button variant="ghost" size="icon" onClick={() => setMobileOpen(false)}>
            <X className="w-5 h-5" />
          </Button>
        </div>
        <div className="flex-1 overflow-hidden">
          {SidebarContent}
        </div>
      </div>
    </>
  )
}
