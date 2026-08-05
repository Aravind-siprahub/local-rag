import { getApiErrorMessage } from '@/api/client'
import { useDeleteDocument } from '@/features/documents/hooks'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import type { DocumentListItem } from '@/types'

interface DeleteDocumentDialogProps {
  target: DocumentListItem | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onDeleted: (documentId: string) => void
}

export function DeleteDocumentDialog({
  target,
  open,
  onOpenChange,
  onDeleted,
}: DeleteDocumentDialogProps) {
  const deleteMutation = useDeleteDocument()

  const handleDelete = async () => {
    if (!target) {
      return
    }

    try {
      await deleteMutation.mutateAsync(target.document.id)
      onDeleted(target.document.id)
      onOpenChange(false)
    } catch {
      // Error surfaced via mutation state below.
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete document?</DialogTitle>
          <DialogDescription>
            This soft-deletes <strong>{target?.document.title}</strong>. Existing versions,
            chunks, and embeddings remain in the database.
          </DialogDescription>
        </DialogHeader>

        {deleteMutation.isError ? (
          <p className="text-sm text-destructive">{getApiErrorMessage(deleteMutation.error)}</p>
        ) : null}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={deleteMutation.isPending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={() => {
              void handleDelete()
            }}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? 'Deleting…' : 'Delete document'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
