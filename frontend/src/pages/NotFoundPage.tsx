import { Link } from 'react-router-dom'

import { buttonVariants } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { ROUTES } from '@/routes/paths'

export function NotFoundPage() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 text-center">
      <p className="text-sm font-medium uppercase tracking-[0.2em] text-muted-foreground">404</p>
      <h1 className="text-4xl font-semibold">Page not found</h1>
      <p className="max-w-md text-muted-foreground">
        The page you are looking for does not exist or has been moved.
      </p>
      <Link to={ROUTES.dashboard} className={cn(buttonVariants())}>
        Back to dashboard
      </Link>
    </div>
  )
}
