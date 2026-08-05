import { useSyncExternalStore } from 'react'

import { settingsStore, type AppSettings } from '@/store'

export function useSettings(): AppSettings {
  return useSyncExternalStore(settingsStore.subscribe, settingsStore.get, settingsStore.get)
}
