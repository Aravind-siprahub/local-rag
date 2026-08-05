import { EyeIcon, MoreHorizontalIcon, Trash2Icon } from 'lucide-react'

import { DocumentPipelineStatusBadge } from '@/features/documents/components/DocumentPipelineStatusBadge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import type { DocumentListItem } from '@/types'
import { formatDateTime, formatFileSize } from '@/utils'

interface DocumentsTableProps {
  items: DocumentListItem[]
  onView: (documentId: string) => void
  onDelete: (item: DocumentListItem) => void
}

export function DocumentsTable({ items, onView, onDelete }: DocumentsTableProps) {
  return (
    <div className="hidden overflow-hidden rounded-xl border border-border/60 md:block">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead>Title</TableHead>
            <TableHead>Filename</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Version</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Size</TableHead>
            <TableHead className="w-12 text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => (
            <TableRow
              key={item.document.id}
              className="cursor-pointer"
              onClick={() => onView(item.document.id)}
            >
              <TableCell className="max-w-[220px]">
                <div className="truncate font-medium">{item.document.title}</div>
                {item.document.description ? (
                  <div className="truncate text-xs text-muted-foreground">
                    {item.document.description}
                  </div>
                ) : null}
              </TableCell>
              <TableCell className="max-w-[180px] truncate text-muted-foreground">
                {item.filename ?? '—'}
              </TableCell>
              <TableCell>
                <DocumentPipelineStatusBadge status={item.displayStatus} />
              </TableCell>
              <TableCell>{item.versionLabel ?? '—'}</TableCell>
              <TableCell className="whitespace-nowrap text-muted-foreground">
                {formatDateTime(item.document.created_at)}
              </TableCell>
              <TableCell className="text-right text-muted-foreground">
                {formatFileSize(item.fileSizeBytes)}
              </TableCell>
              <TableCell className="text-right">
                <DropdownMenu>
                  <DropdownMenuTrigger
                    render={
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Actions for ${item.document.title}`}
                        onClick={(event) => event.stopPropagation()}
                      />
                    }
                  >
                    <MoreHorizontalIcon />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onClick={(event) => {
                        event.stopPropagation()
                        onView(item.document.id)
                      }}
                    >
                      <EyeIcon />
                      View details
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      variant="destructive"
                      onClick={(event) => {
                        event.stopPropagation()
                        onDelete(item)
                      }}
                    >
                      <Trash2Icon />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
