import { Outlet, useLocation } from 'react-router-dom'

import { AppSidebar } from '@/components/AppSidebar'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ROUTES } from '@/routes/paths'

export function AppLayout() {
  const location = useLocation()
  const isChatRoute = location.pathname === ROUTES.chat || location.pathname.startsWith(`${ROUTES.chat}/`)

  return (
    <div className="flex min-h-svh flex-col lg:flex-row">
      <div className="border-b border-border/60 p-3 lg:hidden">
        <AppSidebar />
      </div>

      <div className="hidden lg:fixed lg:inset-y-0 lg:left-0 lg:block lg:w-64 lg:p-4">
        <AppSidebar />
      </div>

      {isChatRoute ? (
        <main className="flex flex-1 flex-col min-h-0 lg:ml-64 lg:h-svh overflow-hidden">
          <Outlet />
        </main>
      ) : (
        <ScrollArea className="flex-1 lg:ml-64">
          <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
            <Outlet />
          </main>
        </ScrollArea>
      )}
    </div>
  )
}
