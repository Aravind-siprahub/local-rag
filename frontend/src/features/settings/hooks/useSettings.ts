import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'

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
    return usersQuery.data?.items.find((u) => u.is_active) ?? usersQuery.data?.items[0] ?? null
  }, [usersQuery.data?.items])

  const updateSettingMutation = useMutation({
    mutationFn: ({ key, value, description }: { key: string; value: unknown; description?: string }) =>
      upsertSystemSetting(key, { value, description, updated_by: activeUser?.email }),
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
