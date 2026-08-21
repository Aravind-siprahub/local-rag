import { Plus, MessageSquare, Trash2, Search, X } from 'lucide-react'
import { useState } from 'react'
import { formatDistanceToNow } from 'date-fns'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import type { Conversation } from '../types/chat'

interface ChatSidebarProps {
  conversations: Conversation[]
  activeId?: string
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  isMobileOpen: boolean
  setMobileOpen: (open: boolean) => void
}

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

  const filtered = conversations.filter(c => 
    c.title.toLowerCase().includes(search.toLowerCase())
  )

  const SidebarContent = (
    <div className="flex flex-col h-full w-full bg-muted/5 border-r border-border/30">
      <div className="p-4 flex flex-col gap-3">
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
      
      <div className="h-px bg-border/20 mx-4 mb-2" />
      
      <ScrollArea className="flex-1">
        <div className="px-2 py-1 flex flex-col gap-0.5">
          {filtered.length === 0 ? (
            <p className="text-[11px] text-muted-foreground/50 text-center p-4">
              No conversations found.
            </p>
          ) : (
            filtered.map(conv => (
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
                    <span className="text-xs truncate max-w-[155px] text-foreground/90">{conv.title}</span>
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
                    e.stopPropagation()
                    onDelete(conv.id)
                  }}
                  title="Delete chat"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
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
