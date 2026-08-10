import React from 'react'

interface FormCardProps {
  children: React.ReactNode
  className?: string
}

export const FormCard: React.FC<FormCardProps> = ({ children, className = '' }) => {
  return (
    <div
      className={`relative w-full max-w-md p-6 sm:p-8 rounded-2xl bg-slate-900/70 border border-slate-800/80 shadow-2xl backdrop-blur-xl transition-all duration-300 hover:border-slate-700/60 ${className}`}
    >
      {/* Subtle top glow highlight */}
      <div className="absolute inset-x-0 -top-px h-px bg-linear-to-r from-transparent via-indigo-500/50 to-transparent rounded-t-2xl pointer-events-none" />

      {/* Card Content */}
      <div className="relative z-10">{children}</div>
    </div>
  )
}
