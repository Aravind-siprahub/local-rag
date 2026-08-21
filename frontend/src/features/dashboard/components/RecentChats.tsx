import { MessageSquareIcon } from 'lucide-react'
import { Link } from 'react-router-dom'

import { EmptyState } from '@/components/EmptyState'
import { ErrorState } from '@/components/ErrorState'
import { LoadingState } from '@/components/LoadingState'
import type { ChatSession } from '@/types'
import { formatRelativeTime } from '@/utils/date'
import { ROUTES } from '@/routes/paths'

interface RecentChatsProps {
  sessions?: ChatSession[]
  isLoading: boolean
  isError: boolean
  error: unknown
  onRetry: () => void
}

export function RecentChats({ sessions, isLoading, isError, error, onRetry }: RecentChatsProps) {
  return (
    <div className="flex flex-col rounded-xl border border-border/50 bg-card shadow-sm overflow-hidden">
      <div className="flex items-center justify-between border-b border-border/50 bg-muted/20 px-5 py-4">
        <h3 className="font-semibold tracking-tight">Recent conversations</h3>
        {!isLoading && !isError && sessions && sessions.length > 0 && (
          <Link
            to={ROUTES.chat}
            className="text-sm font-medium text-primary hover:underline"
          >
            Open chat
          </Link>
        )}
      </div>

      <div className="p-0">
        {isLoading ? <div className="p-5"><LoadingState rows={4} /></div> : null}
        {isError ? <div className="p-5"><ErrorState title="Could not load chat sessions" error={error} onRetry={onRetry} /></div> : null}
        
        {!isLoading && !isError && (sessions ?? []).length === 0 ? (
          <div className="p-8">
            <EmptyState
              title="No conversations yet"
              description="Start a chat to query your documents with local LLM retrieval."
              icon={<MessageSquareIcon className="size-6 text-muted-foreground" />}
            />
          </div>
        ) : null}

        {!isLoading && !isError && sessions && sessions.length > 0 ? (
          <div className="w-full">
            <div className="grid grid-cols-[1fr_auto] items-center gap-4 border-b border-border/50 bg-muted/10 px-5 py-2 text-xs font-medium text-muted-foreground">
              <div>Conversation</div>
              <div className="w-32 text-right">Last activity</div>
            </div>
            <ul className="divide-y divide-border/50">
              {sessions.map((session) => (
                <li
                  key={session.id}
                  className="grid grid-cols-[1fr_auto] items-center gap-4 px-5 py-3 transition-colors hover:bg-muted/10"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <MessageSquareIcon className="size-4 shrink-0 text-muted-foreground" />
                    <p className="truncate text-sm font-medium" title={session.title}>
                      {session.title}
                    </p>
                  </div>
                  <div className="w-32 text-right text-xs text-muted-foreground">
                    {session.last_message_at
                      ? formatRelativeTime(session.last_message_at)
                      : formatRelativeTime(session.created_at)}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  )
}
