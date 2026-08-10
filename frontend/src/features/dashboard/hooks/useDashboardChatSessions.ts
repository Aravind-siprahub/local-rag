import { useQuery } from '@tanstack/react-query'

import { listChatSessions } from '@/services/chat-sessions.service'
import { useCurrentUser } from '@/hooks/useCurrentUser'

const RECENT_CHATS_LIMIT = 5

export function useDashboardChatSessions() {
  const { data: user, isLoading: isUserLoading } = useCurrentUser()

  return useQuery({
    queryKey: ['chat-sessions', 'dashboard', user?.id ?? 'none', RECENT_CHATS_LIMIT],
    queryFn: () =>
      listChatSessions({
        user_id: user?.id,
        limit: RECENT_CHATS_LIMIT,
        offset: 0,
        include_archived: false,
      }),
    // Don't fire until the user query has settled AND we have a real user ID.
    // When user is null (no users exist yet) we skip the API call entirely
    // and fall back to an empty array via the select default.
    enabled: !isUserLoading && !!user?.id,
    select: (response) => response.items,
  })
}
