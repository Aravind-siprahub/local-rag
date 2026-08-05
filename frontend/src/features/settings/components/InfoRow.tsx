import { CopyIcon, CheckIcon } from 'lucide-react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'

interface InfoRowProps {
  label: string
  value: React.ReactNode
  description?: string
  copyable?: boolean
  copyText?: string
}

export function InfoRow({ label, value, description, copyable = false, copyText }: InfoRowProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    const textToCopy = copyText || (typeof value === 'string' ? value : String(value))
    void navigator.clipboard.writeText(textToCopy)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 py-3 border-b border-border/40 last:border-0">
      <div className="space-y-0.5 max-w-sm">
        <span className="text-sm font-medium text-foreground">{label}</span>
        {description ? <p className="text-xs text-muted-foreground">{description}</p> : null}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <div className="text-sm font-mono text-foreground">{value}</div>
        {copyable ? (
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            onClick={handleCopy}
            title="Copy to clipboard"
          >
            {copied ? (
              <CheckIcon className="size-3.5 text-emerald-500" />
            ) : (
              <CopyIcon className="size-3.5 text-muted-foreground" />
            )}
          </Button>
        ) : null}
      </div>
    </div>
  )
}
