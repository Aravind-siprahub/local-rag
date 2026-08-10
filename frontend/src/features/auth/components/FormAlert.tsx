import React, { useState } from 'react'
import { AlertCircle, CheckCircle2, Copy, Check, X } from 'lucide-react'

interface FormAlertProps {
  type?: 'error' | 'success'
  message: string
  onClose?: () => void
}

export const FormAlert: React.FC<FormAlertProps> = ({ type = 'error', message, onClose }) => {
  const [copied, setCopied] = useState(false)

  if (!message) return null

  const handleCopy = () => {
    navigator.clipboard.writeText(message)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const isError = type === 'error'

  return (
    <div
      className={`flex items-start justify-between gap-3 p-3.5 rounded-xl border text-xs leading-relaxed transition-all duration-200 ${
        isError
          ? 'bg-red-500/10 border-red-500/20 text-red-300'
          : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
      }`}
      role="alert"
    >
      <div className="flex items-start gap-2.5">
        {isError ? (
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
        ) : (
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
        )}
        <span className="font-medium">{message}</span>
      </div>

      <div className="flex items-center gap-1 shrink-0">
        <button
          type="button"
          onClick={handleCopy}
          className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-slate-200 transition-colors"
          title="Copy message"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-slate-200 transition-colors"
            title="Dismiss alert"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  )
}
