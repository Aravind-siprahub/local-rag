import { useQuery } from '@tanstack/react-query'

import { listChatSessions } from '@/services/chat-sessions.service'
import { useCurrentUser } from '@/hooks/useCurrentUser'

const RECENT_CHATS_LIMIT = 5

export function useDashboardChatSessions() {
  const { data: user, isLoading: isUserLoading } = useCurrentUser()

  return useQuery({
    queryKey: ['chat-sessions', 'dashboard', user?.id ?? 'default', RECENT_CHATS_LIMIT],
    queryFn: () =>
      listChatSessions({
        user_id: user?.id,
        limit: RECENT_CHATS_LIMIT,
        offset: 0,
        include_archived: false,
      }),
    enabled: !isUserLoading,
    select: (response) => response.items,
  })
}
