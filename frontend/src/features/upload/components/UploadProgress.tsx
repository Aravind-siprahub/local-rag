interface UploadProgressProps {
  overallProgress: number
  totalFiles: number
  completedCount: number
  failedCount: number
  isUploading: boolean
}

export function UploadProgress({
  overallProgress,
  totalFiles,
  completedCount,
  failedCount,
  isUploading,
}: UploadProgressProps) {
  if (totalFiles === 0) return null

  return (
    <div className="p-4 rounded-xl border border-border/60 bg-card/60 space-y-3">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-foreground">Upload Progress</span>
          <span className="text-xs text-muted-foreground">
            ({completedCount} of {totalFiles} completed
            {failedCount > 0 ? `, ${failedCount} failed` : ''})
          </span>
        </div>
        <span className="font-mono text-xs font-semibold text-foreground">{overallProgress}%</span>
      </div>

      <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 rounded-full ${
            overallProgress === 100 ? 'bg-emerald-500' : isUploading ? 'bg-primary' : 'bg-muted-foreground/40'
          }`}
          style={{ width: `${overallProgress}%` }}
        />
      </div>
    </div>
  )
}
