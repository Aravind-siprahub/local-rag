import { Outlet } from 'react-router-dom'

import { AppSidebar } from '@/components/AppSidebar'
import { ScrollArea } from '@/components/ui/scroll-area'

export function AppLayout() {
  return (
    <div className="flex min-h-svh flex-col lg:flex-row">
      <div className="border-b border-border/60 p-3 lg:hidden">
        <AppSidebar />
      </div>

      <div className="hidden lg:fixed lg:inset-y-0 lg:left-0 lg:block lg:w-64 lg:p-4">
        <AppSidebar />
      </div>

      <ScrollArea className="flex-1 lg:ml-64">
        <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          <Outlet />
        </main>
      </ScrollArea>
    </div>
  )
}
