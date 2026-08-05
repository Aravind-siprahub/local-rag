import { AlertCircleIcon } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { getApiErrorMessage } from '@/api/client'

interface ErrorStateProps {
  title?: string
  error: unknown
  onRetry?: () => void
}

export function ErrorState({
  title = 'Something went wrong',
  error,
  onRetry,
}: ErrorStateProps) {
  return (
    <Alert variant="destructive" className="glass-panel border-destructive/30">
      <AlertCircleIcon />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="flex flex-col gap-3">
        <span>{getApiErrorMessage(error)}</span>
        {onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className="w-fit text-sm font-medium underline underline-offset-4"
          >
            Try again
          </button>
        ) : null}
      </AlertDescription>
    </Alert>
  )
}
