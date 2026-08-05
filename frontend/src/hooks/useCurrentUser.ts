import { useQuery } from '@tanstack/react-query'

import { listDocuments } from '@/services/documents.service'
import { listUsers } from '@/services/users.service'
import { settingsStore } from '@/store'
import type { User } from '@/types'

/**
 * Resolve the active app user.
 *
 * Retrieval scopes by chat_session.user_id → Document.user_id. Picking an
 * arbitrary first user (e.g. a Swagger demo account) while documents/embeddings
 * belong to another user yields "Retrieved 0 chunks". Prefer:
 * 1) stored settings.userId
 * 2) the user who owns the most documents
 * 3) first active user
 */
export function useCurrentUser() {
  return useQuery({
    queryKey: ['users', 'current'],
    queryFn: async (): Promise<User | null> => {
      const users = await listUsers({ limit: 100, offset: 0 })
      if (users.items.length === 0) {
        return null
      }

      const documents = await listDocuments({ limit: 100, offset: 0 })
      const ownershipCounts = new Map<string, number>()
      for (const document of documents.items) {
        ownershipCounts.set(
          document.user_id,
          (ownershipCounts.get(document.user_id) ?? 0) + 1,
        )
      }

      const pickOwnerWithMostDocuments = (): User | null => {
        let preferred: User | null = null
        let bestCount = -1
        for (const user of users.items) {
          const count = ownershipCounts.get(user.id) ?? 0
          if (count > bestCount) {
            bestCount = count
            preferred = user
          }
        }
        return preferred
      }

      const ownerPreferred = pickOwnerWithMostDocuments()
      const ownerCount = ownerPreferred ? (ownershipCounts.get(ownerPreferred.id) ?? 0) : 0

      // If a user owns knowledge base documents, ALWAYS use that user as active user
      if (ownerCount > 0 && ownerPreferred) {
        settingsStore.set({ userId: ownerPreferred.id })
        return ownerPreferred
      }

      const storedUserId = settingsStore.get().userId
      if (storedUserId) {
        const matched = users.items.find((user) => user.id === storedUserId)
        if (matched) {
          return matched
        }
      }

      const resolved = users.items[0] ?? null
      if (resolved) {
        settingsStore.set({ userId: resolved.id })
      }
      return resolved
    },
  })
}
