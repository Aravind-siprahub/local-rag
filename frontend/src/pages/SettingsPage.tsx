import { RefreshCwIcon, MoonIcon, SunIcon, MonitorIcon } from 'lucide-react'
import { useState, useEffect } from 'react'

import { TopBar } from '@/components/TopBar'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  HealthBadge,
  InfoRow,
  ReadOnlyValue,
  SettingCard,
  SettingsSection,
  SettingsSidebar,
  useSettings,
} from '@/features/settings'
import type { SettingsSectionId } from '@/types'

export function SettingsPage() {
  const [activeSection, setActiveSection] = useState<SettingsSectionId>('general')
  const [theme, setTheme] = useState<'dark' | 'light' | 'system'>(() => {
    return (localStorage.getItem('theme') as 'dark' | 'light' | 'system') || 'system'
  })

  const { health, isHealthLoading, isHealthError, activeUser, refetchAll } = useSettings()

  useEffect(() => {
    const root = window.document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else if (theme === 'light') {
      root.classList.remove('dark')
    } else {
      const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      if (systemDark) {
        root.classList.add('dark')
      } else {
        root.classList.remove('dark')
      }
    }
    localStorage.setItem('theme', theme)
  }, [theme])

  const backendUrl = import.meta.env.VITE_API_BASE_URL ?? '/api'

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16">
      <TopBar
        title="Settings & System Configuration"
        description="View runtime environment properties, Ollama model configs, vector database health, and theme preferences."
      >
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => {
            void refetchAll()
          }}
          disabled={isHealthLoading}
        >
          <RefreshCwIcon className={`size-3.5 mr-1.5 ${isHealthLoading ? 'animate-spin' : ''}`} />
          Check System Status
        </Button>
      </TopBar>

      <div className="flex flex-col lg:flex-row gap-8">
        {/* Navigation Sidebar */}
        <SettingsSidebar activeSection={activeSection} onSelectSection={setActiveSection} />

        {/* Settings Content Area */}
        <main className="flex-1 space-y-8 min-w-0">
          {/* GENERAL SECTION */}
          {activeSection === 'general' ? (
            <SettingsSection
              id="general"
              title="General Settings"
              description="Application overview, environment information, API endpoints, and active user context."
            >
              <SettingCard
                title="System Overview"
                description="Core application details and running environment."
              >
                <InfoRow label="Application Name" value="Local RAG API" />
                <InfoRow label="Environment" value="development" />
                <InfoRow label="API Base URL" value={backendUrl} copyable />
                <InfoRow
                  label="API Health Status"
                  value={
                    <HealthBadge
                      status={health?.status ?? null}
                      isLoading={isHealthLoading}
                      isError={isHealthError}
                    />
                  }
                />
              </SettingCard>

              <SettingCard
                title="Current User Profile"
                description="Authenticated account context loaded from backend user repository."
              >
                <InfoRow
                  label="User Email"
                  value={activeUser ? activeUser.email : 'Not logged in / Anonymous'}
                  copyable={Boolean(activeUser)}
                />
                <InfoRow
                  label="Role"
                  value={activeUser ? activeUser.role : 'Guest'}
                />
                <InfoRow
                  label="User ID"
                  value={activeUser ? activeUser.id : 'N/A'}
                  copyable={Boolean(activeUser)}
                />
              </SettingCard>
            </SettingsSection>
          ) : null}

          {/* AI SECTION */}
          {activeSection === 'ai' ? (
            <SettingsSection
              id="ai"
              title="AI & LLM Configuration"
              description="Ollama local LLM chat model parameters, temperature, and runtime options."
            >
              <SettingCard
                title="Local Language Model (Ollama)"
                description="Inference configurations enforced by backend settings module."
              >
                <InfoRow
                  label="LLM Provider"
                  value={<Badge variant="secondary" className="font-mono">Ollama (Local)</Badge>}
                />
                <InfoRow label="Chat Model" value={<ReadOnlyValue value="qwen3:8b" />} />
                <InfoRow label="Temperature" value={<ReadOnlyValue value="0.70" />} />
                <InfoRow label="Max Context Tokens" value={<ReadOnlyValue value="2048 tokens" />} />
                <InfoRow
                  label="Streaming Support"
                  value={<Badge variant="outline" className="text-emerald-600 bg-emerald-500/10 border-emerald-500/20">SSE (Server-Sent Events)</Badge>}
                />
                <InfoRow label="LLM Timeout" value={<ReadOnlyValue value="300.0 seconds" />} />
              </SettingCard>
            </SettingsSection>
          ) : null}

          {/* RETRIEVAL SECTION */}
          {activeSection === 'retrieval' ? (
            <SettingsSection
              id="retrieval"
              title="Retrieval & Vector Search"
              description="RAG vector similarity search parameters, top-k candidate limits, and text chunking strategy."
            >
              <SettingCard
                title="Vector Search Parameters"
                description="Controls how relevant document excerpts are retrieved for RAG context."
              >
                <InfoRow label="Top K Candidates" value={<ReadOnlyValue value="10 chunks" />} />
                <InfoRow label="Similarity Threshold" value={<ReadOnlyValue value="0.0 (Return best matching)" />} />
                <InfoRow label="Chunk Size" value={<ReadOnlyValue value="1000 characters" />} />
                <InfoRow label="Chunk Overlap" value={<ReadOnlyValue value="200 characters" />} />
                <InfoRow
                  label="Search Strategy"
                  value={<Badge variant="secondary">pgvector Cosine Distance (&lt;=&gt;)</Badge>}
                />
              </SettingCard>
            </SettingsSection>
          ) : null}

          {/* EMBEDDINGS SECTION */}
          {activeSection === 'embeddings' ? (
            <SettingsSection
              id="embeddings"
              title="Embeddings Configuration"
              description="Dense vector representation model details and PostgreSQL vector column dimensions."
            >
              <SettingCard
                title="Embedding Model & Storage"
                description="Model specification used to generate text embeddings."
              >
                <InfoRow label="Embedding Model" value={<ReadOnlyValue value="nomic-embed-text" />} />
                <InfoRow label="Vector Dimensions" value={<ReadOnlyValue value="768" />} />
                <InfoRow label="Vector Database" value={<ReadOnlyValue value="PostgreSQL + pgvector Extension" />} />
                <InfoRow
                  label="Embedding Status"
                  value={<Badge variant="secondary" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/20">Active & Ready</Badge>}
                />
              </SettingCard>
            </SettingsSection>
          ) : null}

          {/* APPEARANCE SECTION */}
          {activeSection === 'appearance' ? (
            <SettingsSection
              id="appearance"
              title="Appearance & Interface"
              description="Customize the visual theme and interface preferences."
            >
              <SettingCard
                title="Theme Preference"
                description="Switch between light, dark, or system default color scheme."
              >
                <div className="py-2 grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <button
                    type="button"
                    onClick={() => setTheme('light')}
                    className={`
                      p-4 rounded-lg border flex flex-col items-center gap-2 text-xs font-medium transition-all
                      ${
                        theme === 'light'
                          ? 'border-primary bg-primary/5 text-primary ring-2 ring-primary/20'
                          : 'border-border/60 bg-card hover:bg-accent/40 text-muted-foreground'
                      }
                    `}
                  >
                    <SunIcon className="size-5" />
                    <span>Light Mode</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setTheme('dark')}
                    className={`
                      p-4 rounded-lg border flex flex-col items-center gap-2 text-xs font-medium transition-all
                      ${
                        theme === 'dark'
                          ? 'border-primary bg-primary/5 text-primary ring-2 ring-primary/20'
                          : 'border-border/60 bg-card hover:bg-accent/40 text-muted-foreground'
                      }
                    `}
                  >
                    <MoonIcon className="size-5" />
                    <span>Dark Mode</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setTheme('system')}
                    className={`
                      p-4 rounded-lg border flex flex-col items-center gap-2 text-xs font-medium transition-all
                      ${
                        theme === 'system'
                          ? 'border-primary bg-primary/5 text-primary ring-2 ring-primary/20'
                          : 'border-border/60 bg-card hover:bg-accent/40 text-muted-foreground'
                      }
                    `}
                  >
                    <MonitorIcon className="size-5" />
                    <span>System Default</span>
                  </button>
                </div>
              </SettingCard>
            </SettingsSection>
          ) : null}

          {/* SYSTEM SECTION */}
          {activeSection === 'system' ? (
            <SettingsSection
              id="system"
              title="System & Infrastructure"
              description="Health checks for database, Ollama execution host, and local disk storage."
            >
              <SettingCard
                title="Service Health"
                description="Live connectivity diagnostics across system dependencies."
              >
                <InfoRow
                  label="Backend API Process"
                  value={
                    <HealthBadge
                      status={health?.status ?? null}
                      isLoading={isHealthLoading}
                      isError={isHealthError}
                    />
                  }
                />
                <InfoRow
                  label="Database (PostgreSQL)"
                  value={
                    health?.database === 'connected' ? (
                      <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/20">Connected</Badge>
                    ) : (
                      <Badge variant="destructive">Disconnected</Badge>
                    )
                  }
                />
                <InfoRow label="Local File Storage" value={<ReadOnlyValue value="./uploads directory" />} />
                <InfoRow label="Execution Mode" value={<ReadOnlyValue value="GPU (Ollama auto-offload)" />} />
              </SettingCard>
            </SettingsSection>
          ) : null}

          {/* ABOUT SECTION */}
          {activeSection === 'about' ? (
            <SettingsSection
              id="about"
              title="About Local RAG"
              description="Software versioning, framework dependencies, and build details."
            >
              <SettingCard
                title="Version Information"
                description="Full build versions across local stack components."
              >
                <InfoRow label="Application Version" value="1.0.0" />
                <InfoRow label="Frontend Package" value="1.0.0" />
                <InfoRow label="Backend API Version" value="1.0.0" />
                <InfoRow label="React Framework" value="19.2.8" />
                <InfoRow label="Tailwind CSS" value="v4.3.3" />
                <InfoRow label="Build Timestamp" value={new Date().toISOString().slice(0, 10)} />
              </SettingCard>
            </SettingsSection>
          ) : null}
        </main>
      </div>
    </div>
  )
}
