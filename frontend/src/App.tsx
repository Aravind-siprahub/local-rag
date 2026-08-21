import { useEffect } from 'react'
import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from 'react-router-dom'

import { queryClient } from '@/lib/query-client'
import { appRouter } from '@/routes'
import { settingsStore, applyTheme } from '@/store/settings.store'

export function App() {
  useEffect(() => {
    // Initial sync
    const currentTheme = settingsStore.get().theme
    applyTheme(currentTheme)

    // Listen to store updates (updates when changed in settings page)
    const unsubscribe = settingsStore.subscribe(() => {
      const updated = settingsStore.get()
      applyTheme(updated.theme)
    })

    // Listen to system prefers-color-scheme changes reactively
    const systemQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleSystemChange = () => {
      const current = settingsStore.get()
      if (current.theme === 'system') {
        applyTheme('system')
      }
    }

    systemQuery.addEventListener('change', handleSystemChange)
    
    return () => {
      unsubscribe()
      systemQuery.removeEventListener('change', handleSystemChange)
    }
  }, [])

  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={appRouter} />
    </QueryClientProvider>
  )
}
