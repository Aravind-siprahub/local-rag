import { MessageSquareIcon } from 'lucide-react'
import { Link } from 'react-router-dom'

import { EmptyState } from '@/components/EmptyState'
import { ErrorState } from '@/components/ErrorState'
import { LoadingState } from '@/components/LoadingState'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
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
    <Card className="glass-panel border-border/60">
      <CardHeader>
        <CardTitle>Recent chats</CardTitle>
        <CardDescription>Your latest RAG conversations.</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? <LoadingState rows={5} /> : null}
        {isError ? <ErrorState title="Could not load chat sessions" error={error} onRetry={onRetry} /> : null}
        {!isLoading && !isError && sessions?.length === 0 ? (
          <EmptyState
            title="No conversations yet"
            description="Start a chat to query your documents with local LLM retrieval."
            icon={<MessageSquareIcon className="size-5" />}
          />
        ) : null}
        {!isLoading && !isError && sessions && sessions.length > 0 ? (
          <ul className="space-y-3">
            {sessions.map((session) => (
              <li
                key={session.id}
                className="flex items-start justify-between gap-4 rounded-xl border border-border/60 bg-background/40 px-4 py-3 transition-colors hover:border-secondary/30"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-foreground">{session.title}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {session.last_message_at
                      ? `Last message ${formatRelativeTime(session.last_message_at)}`
                      : `Created ${formatRelativeTime(session.created_at)}`}
                  </p>
                </div>
                <Badge variant="outline" className="shrink-0 border-secondary/30 bg-secondary/10 text-secondary">
                  Session
                </Badge>
              </li>
            ))}
          </ul>
        ) : null}

        {!isLoading && !isError && sessions && sessions.length > 0 ? (
          <Link
            to={ROUTES.chat}
            className="mt-4 inline-flex text-sm font-medium text-secondary hover:underline"
          >
            Open chat
          </Link>
        ) : null}
      </CardContent>
    </Card>
  )
}
