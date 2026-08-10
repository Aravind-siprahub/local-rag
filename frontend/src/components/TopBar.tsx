import { ActivityIcon, LogOutIcon } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import { useAuth } from '@/features/auth/hooks/authHooks'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useHealth } from '@/hooks/useHealth'

import type { ReactNode } from 'react'

interface TopBarProps {
  title: string
  description?: string
  children?: ReactNode
}

export function TopBar({ title, description, children }: TopBarProps) {
  const navigate = useNavigate()
  const { logout, isAuthenticated } = useAuth()
  const { data: health, isLoading, isError } = useHealth()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  const statusLabel = isLoading ? 'Checking…' : isError ? 'Offline' : 'Online'
  const statusClass = isError
    ? 'border-destructive/30 bg-destructive/10 text-destructive'
    : 'border-success/30 bg-success/10 text-[hsl(var(--success))]'

  return (
    <header className="flex flex-col gap-4 border-b border-border/60 pb-6 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <p className="text-sm font-medium text-muted-foreground">Overview</p>
        <h2 className="mt-1 text-3xl font-semibold tracking-tight">{title}</h2>
        {description ? (
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">{description}</p>
        ) : null}
      </div>

      <div className="flex items-center gap-3 shrink-0 flex-wrap">
        {children}
        <Badge variant="outline" className={`gap-1.5 px-3 py-1 ${statusClass}`}>
          <ActivityIcon className="size-3.5" />
          API {statusLabel}
          {health?.database === 'connected' ? ' · DB connected' : null}
        </Badge>
        {isAuthenticated ? (
          <Button
            variant="outline"
            size="sm"
            onClick={handleLogout}
            className="gap-1.5 border-border/60 text-muted-foreground hover:text-red-400 hover:border-red-500/30"
          >
            <LogOutIcon className="size-3.5" />
            Sign Out
          </Button>
        ) : null}
      </div>
    </header>
  )
}
