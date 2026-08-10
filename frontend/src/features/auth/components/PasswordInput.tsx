import React, { useState, forwardRef } from 'react'
import { Eye, EyeOff, Lock } from 'lucide-react'

export interface PasswordInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: string
  label?: string
}

export const PasswordInput = forwardRef<HTMLInputElement, PasswordInputProps>(
  ({ error, label, className = '', ...props }, ref) => {
    const [showPassword, setShowPassword] = useState(false)

    return (
      <div className="w-full space-y-1.5">
        {label && (
          <label
            htmlFor={props.id || props.name}
            className="block text-xs font-medium text-slate-300 tracking-wide"
          >
            {label}
          </label>
        )}
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
            <Lock className="w-4 h-4" />
          </div>
          <input
            ref={ref}
            type={showPassword ? 'text' : 'password'}
            className={`w-full pl-10 pr-10 py-2.5 bg-slate-950/60 border ${
              error ? 'border-red-500/80 focus:ring-red-500/30' : 'border-slate-800 focus:border-indigo-500 focus:ring-indigo-500/20'
            } rounded-xl text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 transition-all duration-200 ${className}`}
            {...props}
          />
          <button
            type="button"
            onClick={() => setShowPassword((prev) => !prev)}
            tabIndex={-1}
            className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-slate-400 hover:text-slate-200 transition-colors focus:outline-none"
            aria-label={showPassword ? 'Hide password' : 'Show password'}
          >
            {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
        {error && <p className="text-xs text-red-400 mt-1 font-medium">{error}</p>}
      </div>
    )
  }
)

PasswordInput.displayName = 'PasswordInput'
