import { useQuery } from '@tanstack/react-query'

import { listDocuments } from '@/services/documents.service'
import { createUser, listUsers } from '@/services/users.service'
import { settingsStore } from '@/store'
import { AuthStore } from '@/features/auth/utils/authStore'
import type { User } from '@/types'

/**
 * Resolve the active app user.
 *
 * Priority:
 * 1) Find a DB user matching the currently logged-in auth email
 * 2) The user who owns the most documents
 * 3) The stored settings.userId user
 * 4) The first active user
 * 5) Auto-provision a new user if DB has none (no more "No active user" banner)
 */
export function useCurrentUser() {
  return useQuery({
    queryKey: ['users', 'current'],
    queryFn: async (): Promise<User | null> => {
      // ── 1. Fetch all DB users ──────────────────────────────────────────────
      const users = await listUsers({ limit: 100, offset: 0 })

      // ── 2. Auto-provision if DB has no users yet ───────────────────────────
      if (users.items.length === 0) {
        const authUser = AuthStore.getUser()
        try {
          const created = await createUser({
            email: authUser?.email ?? 'admin@localrag.internal',
            full_name: authUser?.fullName ?? 'Admin',
            // Must satisfy backend PasswordPolicy:
            // ≥8 chars, uppercase, lowercase, digit, special character
            password: 'LocalRag@Default1',
            role: 'member',
          })
          settingsStore.set({ userId: created.id })
          return created
        } catch (err: unknown) {
          // Log the exact 422 detail so it's easy to debug
          const e = err as { response?: { data?: unknown; status?: number } }
          if (e?.response) {
            console.error(
              '[useCurrentUser] Failed to auto-provision user:',
              e.response.status,
              JSON.stringify(e.response.data, null, 2),
            )
          } else {
            console.error('[useCurrentUser] Failed to auto-provision user:', err)
          }
          return null
        }
      }

      // ── 3. If auth user is logged in, prefer matching DB user by email ─────
      const authUser = AuthStore.getUser()
      if (authUser?.email) {
        const matched = users.items.find(
          (u) => u.email.toLowerCase() === authUser.email.toLowerCase(),
        )
        if (matched) {
          settingsStore.set({ userId: matched.id })
          return matched
        }
      }

      // ── 4. Prefer the user owning the most documents ───────────────────────
      const documents = await listDocuments({ limit: 100, offset: 0 })
      const ownershipCounts = new Map<string, number>()
      for (const doc of documents.items) {
        ownershipCounts.set(doc.user_id, (ownershipCounts.get(doc.user_id) ?? 0) + 1)
      }

      let topUser: User | null = null
      let topCount = -1
      for (const user of users.items) {
        const count = ownershipCounts.get(user.id) ?? 0
        if (count > topCount) {
          topCount = count
          topUser = user
        }
      }

      if (topCount > 0 && topUser) {
        settingsStore.set({ userId: topUser.id })
        return topUser
      }

      // ── 5. Stored settings userId ──────────────────────────────────────────
      const storedUserId = settingsStore.get().userId
      if (storedUserId) {
        const stored = users.items.find((u) => u.id === storedUserId)
        if (stored) return stored
      }

      // ── 6. First available user ────────────────────────────────────────────
      const first = users.items[0] ?? null
      if (first) settingsStore.set({ userId: first.id })
      return first
    },
  })
}
