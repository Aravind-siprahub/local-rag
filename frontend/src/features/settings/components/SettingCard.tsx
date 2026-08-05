import type { ReactNode } from 'react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface SettingCardProps {
  title: string
  description?: string
  children: ReactNode
  action?: ReactNode
}

export function SettingCard({ title, description, children, action }: SettingCardProps) {
  return (
    <Card className="border border-border/60 bg-card shadow-2xs">
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-4 border-b border-border/30">
        <div className="space-y-1">
          <CardTitle className="text-base font-semibold text-foreground">{title}</CardTitle>
          {description ? <CardDescription className="text-xs">{description}</CardDescription> : null}
        </div>
        {action ? <div>{action}</div> : null}
      </CardHeader>
      <CardContent className="pt-4 space-y-1">{children}</CardContent>
    </Card>
  )
}
