import {
  FileTextIcon,
  LayoutDashboardIcon,
  MessageSquareIcon,
  SettingsIcon,
  UploadIcon,
} from 'lucide-react'
import { NavLink } from 'react-router-dom'

import { cn } from '@/lib/utils'
import { ROUTES } from '@/routes/paths'

const navItems = [
  { to: ROUTES.dashboard, label: 'Dashboard', icon: LayoutDashboardIcon },
  { to: ROUTES.documents, label: 'Documents', icon: FileTextIcon },
  { to: ROUTES.upload, label: 'Upload', icon: UploadIcon },
  { to: ROUTES.chat, label: 'Chat', icon: MessageSquareIcon },
  { to: ROUTES.settings, label: 'Settings', icon: SettingsIcon },
] as const

export function AppSidebar() {
  return (
    <aside className="glass-panel flex h-full w-full flex-col gap-6 p-4 lg:w-64">
      <div className="px-2 pt-2">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          Local RAG
        </p>
        <h1 className="mt-2 text-xl font-semibold gradient-text">Knowledge Studio</h1>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === ROUTES.dashboard}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all',
                isActive
                  ? 'bg-primary/15 text-primary shadow-[inset_0_0_0_1px_hsl(var(--primary)/0.25)]'
                  : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
              )
            }
          >
            <Icon className="size-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="rounded-lg border border-border/60 bg-muted/20 px-3 py-3 text-xs text-muted-foreground">
        Connected to your local RAG backend via API proxy.
      </div>
    </aside>
  )
}
