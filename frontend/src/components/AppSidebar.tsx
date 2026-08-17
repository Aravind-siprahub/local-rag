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
    <aside className="glass-panel flex h-full w-full flex-col gap-5 p-4 lg:w-64">
      <div className="px-2 pt-1">
        <p className="text-[10px] font-bold uppercase tracking-[0.25em] text-primary/80">
          Local RAG
        </p>
        <h1 className="mt-1 text-lg font-bold tracking-tight text-foreground font-display">
          Knowledge Studio
        </h1>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === ROUTES.dashboard}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 ease-in-out',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground',
              )
            }
          >
            <Icon className="size-4 shrink-0 transition-transform duration-200 group-hover:scale-105" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* User Profile & Sign Out Section */}
      <div className="space-y-3 pt-3 border-t border-border/30">
        <div className="flex items-center justify-between gap-2 px-1">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center font-bold text-xs shrink-0 border border-primary/20">
              {user?.fullName ? user.fullName[0].toUpperCase() : <UserIcon className="w-4 h-4" />}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold text-foreground/90 truncate leading-snug">{user?.fullName || 'User'}</p>
              <p className="text-[10px] text-muted-foreground/60 truncate leading-none mt-0.5">{user?.email || 'Authenticated'}</p>
            </div>
          </div>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={handleLogout}
          className="w-full justify-start text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 gap-2 font-medium h-8 rounded-md transition-colors"
        >
          <LogOutIcon className="w-3.5 h-3.5" />
          Sign Out
        </Button>
      </div>

      <div className="rounded-lg border border-border/30 bg-muted/10 px-3 py-2 text-[10px] text-muted-foreground/50 leading-relaxed font-mono">
        Connected to local RAG backend via API proxy.
      </div>
    </aside>
  )
}
