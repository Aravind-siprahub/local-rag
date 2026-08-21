import { MessageSquarePlusIcon, UploadCloudIcon } from 'lucide-react'
import { Link } from 'react-router-dom'

import { ROUTES } from '@/routes/paths'

export function QuickActions() {
  return (
    <div className="flex flex-col rounded-xl border border-border/50 bg-card shadow-sm overflow-hidden h-full">
      <div className="border-b border-border/50 bg-muted/20 px-5 py-4">
        <h3 className="font-semibold tracking-tight">What do you want to do?</h3>
      </div>
      <div className="flex flex-col sm:flex-row gap-4 p-5">
        <Link
          to={ROUTES.chat}
          className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-primary px-4 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
        >
          <MessageSquarePlusIcon className="size-4" />
          Ask your knowledge base
        </Link>
        <Link
          to={ROUTES.upload}
          className="flex flex-1 items-center justify-center gap-2 rounded-lg border border-input bg-background px-4 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          <UploadCloudIcon className="size-4" />
          Upload Document
        </Link>
      </div>
    </div>
  )
}
