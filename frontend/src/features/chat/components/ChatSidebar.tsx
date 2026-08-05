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
    <div className="flex flex-col h-full w-full bg-muted/20 border-r">
      <div className="p-4 flex flex-col gap-4">
        <Button onClick={onNew} className="w-full justify-start" variant="default">
          <Plus className="w-4 h-4 mr-2" />
          New Chat
        </Button>
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search conversations..."
            className="pl-9 bg-background"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="p-2 flex flex-col gap-1">
          {filtered.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center p-4">
              No conversations found.
            </p>
          ) : (
            filtered.map(conv => (
              <div
                key={conv.id}
                className={cn(
                  "group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-colors",
                  activeId === conv.id ? "bg-accent text-accent-foreground" : "hover:bg-muted"
                )}
                onClick={() => {
                  onSelect(conv.id)
                  setMobileOpen(false)
                }}
              >
                <div className="flex items-center min-w-0 overflow-hidden">
                  <MessageSquare className="w-4 h-4 mr-3 shrink-0 opacity-70" />
                  <div className="flex flex-col truncate">
                    <span className="text-sm font-medium truncate">{conv.title}</span>
                    {conv.last_message_at && (
                      <span className="text-xs text-muted-foreground truncate">
                        {formatDistanceToNow(new Date(conv.last_message_at), { addSuffix: true })}
                      </span>
                    )}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity",
                    activeId === conv.id ? "opacity-100 text-accent-foreground" : "text-muted-foreground"
                  )}
                  onClick={(e) => {
                    e.stopPropagation()
                    onDelete(conv.id)
                  }}
                  title="Delete chat"
                >
                  <Trash2 className="w-4 h-4" />
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
