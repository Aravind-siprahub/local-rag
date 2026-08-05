import { TopBar } from '@/components/TopBar'

export function PlaceholderPage({
  title,
  description,
}: {
  title: string
  description: string
}) {
  return (
    <div className="space-y-6">
      <TopBar title={title} description={description} />
      <div className="glass-panel rounded-xl border border-dashed border-border/70 px-6 py-16 text-center text-muted-foreground">
        Coming in the next milestone.
      </div>
    </div>
  )
}
