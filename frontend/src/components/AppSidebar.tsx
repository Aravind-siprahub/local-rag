import {
  FileTextIcon,
  LayoutDashboardIcon,
  LogOutIcon,
  MessageSquareIcon,
  SettingsIcon,
  UploadIcon,
  UserIcon,
} from 'lucide-react'
import { NavLink, useNavigate } from 'react-router-dom'

import { useAuth } from '@/features/auth/hooks/authHooks'
import { Button } from '@/components/ui/button'
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
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

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

      {/* User Profile & Sign Out Section */}
      <div className="space-y-3 pt-2 border-t border-border/40">
        <div className="flex items-center justify-between gap-2 px-1">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-xs shrink-0 border border-primary/30">
              {user?.fullName ? user.fullName[0].toUpperCase() : <UserIcon className="w-4 h-4" />}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold text-foreground truncate">{user?.fullName || 'User'}</p>
              <p className="text-[10px] text-muted-foreground truncate">{user?.email || 'Authenticated'}</p>
            </div>
          </div>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={handleLogout}
          className="w-full justify-start text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 gap-2 font-medium h-9"
        >
          <LogOutIcon className="w-3.5 h-3.5" />
          Sign Out
        </Button>
      </div>

      <div className="rounded-lg border border-border/60 bg-muted/20 px-3 py-2.5 text-[11px] text-muted-foreground">
        Connected to local RAG backend via API proxy.
      </div>
    </aside>
  )
}
