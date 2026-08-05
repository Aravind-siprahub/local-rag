import {
  SlidersIcon,
  BotIcon,
  SearchIcon,
  LayersIcon,
  PaletteIcon,
  ServerIcon,
  InfoIcon,
} from 'lucide-react'

import type { SettingsNavItem, SettingsSectionId } from '@/types'

export const SETTINGS_NAV_ITEMS: (SettingsNavItem & { icon: React.ComponentType<{ className?: string }> })[] = [
  {
    id: 'general',
    label: 'General',
    description: 'Application overview, API base URL & health',
    icon: SlidersIcon,
  },
  {
    id: 'ai',
    label: 'AI & LLM',
    description: 'Chat model, provider & inference parameters',
    icon: BotIcon,
  },
  {
    id: 'retrieval',
    label: 'Retrieval',
    description: 'Vector search strategy, Top-K & chunk sizes',
    icon: SearchIcon,
  },
  {
    id: 'embeddings',
    label: 'Embeddings',
    description: 'Embedding model, dimensions & pgvector status',
    icon: LayersIcon,
  },
  {
    id: 'appearance',
    label: 'Appearance',
    description: 'Theme customization and UI density',
    icon: PaletteIcon,
  },
  {
    id: 'system',
    label: 'System & Services',
    description: 'PostgreSQL, Ollama & file storage health',
    icon: ServerIcon,
  },
  {
    id: 'about',
    label: 'About',
    description: 'Version details, frameworks & build info',
    icon: InfoIcon,
  },
]

interface SettingsSidebarProps {
  activeSection: SettingsSectionId
  onSelectSection: (id: SettingsSectionId) => void
}

export function SettingsSidebar({ activeSection, onSelectSection }: SettingsSidebarProps) {
  return (
    <nav aria-label="Settings navigation" className="w-full lg:w-64 shrink-0">
      {/* Desktop Vertical Navigation */}
      <div className="hidden lg:flex flex-col space-y-1 sticky top-6">
        <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Settings Categories
        </h3>
        {SETTINGS_NAV_ITEMS.map((item) => {
          const Icon = item.icon
          const isActive = activeSection === item.id

          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectSection(item.id)}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-left transition-all duration-150
                ${
                  isActive
                    ? 'bg-primary text-primary-foreground shadow-2xs'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/60'
                }
              `}
            >
              <Icon className={`size-4 shrink-0 ${isActive ? 'text-primary-foreground' : 'text-muted-foreground'}`} />
              <div className="min-w-0">
                <div className="truncate">{item.label}</div>
              </div>
            </button>
          )
        })}
      </div>

      {/* Tablet & Mobile Horizontal Tabs */}
      <div className="lg:hidden flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-border/40 scrollbar-none">
        {SETTINGS_NAV_ITEMS.map((item) => {
          const Icon = item.icon
          const isActive = activeSection === item.id

          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectSection(item.id)}
              className={`
                flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium shrink-0 transition-colors
                ${
                  isActive
                    ? 'bg-primary text-primary-foreground font-semibold'
                    : 'bg-muted/40 text-muted-foreground hover:text-foreground hover:bg-muted'
                }
              `}
            >
              <Icon className="size-3.5" />
              <span>{item.label}</span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}
