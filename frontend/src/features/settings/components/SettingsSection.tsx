import type { ReactNode } from 'react'

interface SettingsSectionProps {
  id: string
  title: string
  description: string
  children: ReactNode
}

export function SettingsSection({ id, title, description, children }: SettingsSectionProps) {
  return (
    <section id={id} className="space-y-4 scroll-mt-6">
      <div className="pb-2 border-b border-border/40">
        <h2 className="text-xl font-bold tracking-tight text-foreground">{title}</h2>
        <p className="text-sm text-muted-foreground mt-0.5">{description}</p>
      </div>
      <div className="space-y-4">{children}</div>
    </section>
  )
}
