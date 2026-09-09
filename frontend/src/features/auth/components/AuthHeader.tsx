import React from 'react'
import { Sparkles } from 'lucide-react'

interface AuthHeaderProps {
  title: string
  subtitle: string
}

export const AuthHeader: React.FC<AuthHeaderProps> = ({ title, subtitle }) => {
  return (
    <div className="text-center mb-8">
      {/* SaaS Brand Logo */}
      <div className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-semibold uppercase tracking-wider mb-5 backdrop-blur-md">
        <Sparkles className="w-3.5 h-3.5 text-primary animate-pulse" />
        <span>Talk to My Data</span>
      </div>

      {/* Main Title */}
      <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground mb-2 font-sans">
        {title}
      </h1>

      {/* Subtitle */}
      <p className="text-sm text-muted-foreground max-w-sm mx-auto leading-relaxed">
        {subtitle}
      </p>
    </div>
  )
}
