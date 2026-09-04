import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'

import { AuthStore } from '@/features/auth/utils/authStore'
import { getHealth, listSystemSettings, listUsers, upsertSystemSetting } from '@/services'
import type { SystemSetting } from '@/types'

export function useSettings() {
  const queryClient = useQueryClient()

  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30_000,
  })

  const usersQuery = useQuery({
    queryKey: ['users'],
    queryFn: () => listUsers({ limit: 10, offset: 0 }),
  })

  const settingsQuery = useQuery({
    queryKey: ['system-settings'],
    queryFn: () => listSystemSettings({ limit: 100, offset: 0 }),
  })

  const settingsMap = useMemo(() => {
    const map = new Map<string, SystemSetting>()
    if (settingsQuery.data?.items) {
      for (const setting of settingsQuery.data.items) {
        map.set(setting.key, setting)
      }
    }
    return map
  }, [settingsQuery.data?.items])

  const activeUser = useMemo(() => {
    const authUser = AuthStore.getUser()
    if (authUser?.email && usersQuery.data?.items) {
      const match = usersQuery.data.items.find(
        (u) => u.email.toLowerCase() === authUser.email.toLowerCase(),
      )
      if (match) return match
    }
    return usersQuery.data?.items.find((u) => u.is_active) ?? usersQuery.data?.items[0] ?? null
  }, [usersQuery.data?.items])

  const updateSettingMutation = useMutation({
    mutationFn: ({ key, value, description }: { key: string; value: unknown; description?: string }) => {
      const formattedValue =
        typeof value === 'object' && value !== null && !Array.isArray(value)
          ? (value as Record<string, unknown>)
          : { val: value }
      const validUserId = activeUser?.id && /^[0-9a-fA-F-]{36}$/.test(activeUser.id) ? activeUser.id : undefined

      return upsertSystemSetting(key, {
        value: formattedValue,
        description,
        updated_by: validUserId,
      })
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['system-settings'] })
    },
  })

  return {
    health: healthQuery.data ?? null,
    isHealthLoading: healthQuery.isLoading,
    isHealthError: healthQuery.isError,
    activeUser,
    settingsMap,
    isSettingsLoading: settingsQuery.isLoading,
    isSettingsError: settingsQuery.isError,
    updateSetting: updateSettingMutation.mutateAsync,
    isUpdating: updateSettingMutation.isPending,
    refetchAll: async () => {
      await Promise.all([healthQuery.refetch(), usersQuery.refetch(), settingsQuery.refetch()])
    },
  }
}
