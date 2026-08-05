import { AlertTriangleIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import type { RejectedFile } from '@/types'
import { formatFileSize } from '@/utils'

interface UnsupportedFileDialogProps {
  rejectedFiles: RejectedFile[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onClear: () => void
}

export function UnsupportedFileDialog({
  rejectedFiles,
  open,
  onOpenChange,
  onClear,
}: UnsupportedFileDialogProps) {
  if (rejectedFiles.length === 0) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md sm:max-w-lg">
        <DialogHeader>
          <div className="flex items-center gap-2 text-destructive">
            <AlertTriangleIcon className="size-5 shrink-0" />
            <DialogTitle>Could Not Add Some Files</DialogTitle>
          </div>
          <DialogDescription>
            {rejectedFiles.length === 1
              ? '1 file was skipped because it did not meet upload requirements.'
              : `${rejectedFiles.length} files were skipped because they did not meet upload requirements.`}
          </DialogDescription>
        </DialogHeader>

        <div className="max-h-60 overflow-y-auto space-y-2 py-2 pr-1">
          {rejectedFiles.map((rf, idx) => (
            <div
              key={`${rf.file.name}-${idx}`}
              className="p-3 rounded-md bg-destructive/5 border border-destructive/20 text-xs space-y-1"
            >
              <div className="flex items-center justify-between font-medium text-foreground">
                <span className="truncate max-w-60">{rf.file.name}</span>
                <span className="font-mono text-muted-foreground">
                  {formatFileSize(rf.file.size)}
                </span>
              </div>
              <p className="text-destructive font-medium">{rf.reason}</p>
            </div>
          ))}
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="secondary"
            onClick={() => {
              onClear()
              onOpenChange(false)
            }}
          >
            Acknowledge & Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
