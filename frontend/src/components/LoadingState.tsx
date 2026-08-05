import { Skeleton } from '@/components/ui/skeleton'

interface LoadingStateProps {
  rows?: number
}

export function LoadingState({ rows = 3 }: LoadingStateProps) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, index) => (
        <Skeleton key={index} className="h-16 w-full rounded-xl" />
      ))}
    </div>
  )
}
