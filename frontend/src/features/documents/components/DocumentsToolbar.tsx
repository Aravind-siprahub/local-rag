import { RefreshCwIcon, SearchIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface DocumentsToolbarProps {
  searchQuery: string
  onSearchChange: (value: string) => void
  onRefresh: () => void
  isRefreshing: boolean
  totalCount: number
  filteredCount: number
}

export function DocumentsToolbar({
  searchQuery,
  onSearchChange,
  onRefresh,
  isRefreshing,
  totalCount,
  filteredCount,
}: DocumentsToolbarProps) {
  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
      <div className="relative w-full lg:max-w-md">
        <SearchIcon className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={searchQuery}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder="Search by title or filename…"
          className="glass-panel border-border/60 bg-background/40 pl-9"
          aria-label="Search documents"
        />
      </div>

      <div className="flex items-center justify-between gap-3 lg:justify-end">
        <p className="text-sm text-muted-foreground">
          Showing {filteredCount} of {totalCount}
        </p>
        <Button
          type="button"
          variant="outline"
          onClick={onRefresh}
          disabled={isRefreshing}
          className="glass-panel-hover"
        >
          <RefreshCwIcon className={isRefreshing ? 'animate-spin' : undefined} />
          Refresh
        </Button>
      </div>
    </div>
  )
}
