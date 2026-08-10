import React from 'react'
import { Check, X } from 'lucide-react'

interface PasswordStrengthMeterProps {
  password: string
}

interface Criterion {
  label: string
  met: boolean
}

export const PasswordStrengthMeter: React.FC<PasswordStrengthMeterProps> = ({ password }) => {
  if (!password) return null

  const criteria: Criterion[] = [
    { label: 'At least 8 characters', met: password.length >= 8 },
    { label: 'Contains uppercase letter', met: /[A-Z]/.test(password) },
    { label: 'Contains lowercase letter', met: /[a-z]/.test(password) },
    { label: 'Contains a number', met: /[0-9]/.test(password) },
    { label: 'Contains special character (@$!%*?&)', met: /[^A-Za-z0-9]/.test(password) },
  ]

  const score = criteria.filter((c) => c.met).length

  const getStrengthLabel = () => {
    if (score <= 1) return { label: 'Weak', color: 'bg-red-500', text: 'text-red-400' }
    if (score <= 3) return { label: 'Fair', color: 'bg-amber-500', text: 'text-amber-400' }
    if (score === 4) return { label: 'Good', color: 'bg-blue-500', text: 'text-blue-400' }
    return { label: 'Strong', color: 'bg-emerald-500', text: 'text-emerald-400' }
  }

  const { label, color, text } = getStrengthLabel()
  const percentage = (score / criteria.length) * 100

  return (
    <div className="space-y-3 mt-2 p-3 bg-slate-950/40 rounded-xl border border-slate-850/60">
      {/* Strength Bar */}
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400 font-medium">Password strength</span>
        <span className={`font-semibold ${text}`}>{label}</span>
      </div>

      <div className="w-full bg-slate-800/60 rounded-full h-1.5 overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {/* Criteria Checklist */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 pt-1">
        {criteria.map((item, idx) => (
          <div key={idx} className="flex items-center gap-1.5 text-[11px]">
            {item.met ? (
              <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
            ) : (
              <X className="w-3.5 h-3.5 text-slate-500 shrink-0" />
            )}
            <span className={item.met ? 'text-slate-300' : 'text-slate-500'}>
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
