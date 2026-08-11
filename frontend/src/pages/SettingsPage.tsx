import { RefreshCwIcon, MoonIcon, SunIcon, MonitorIcon, SaveIcon, CheckIcon, RotateCcwIcon, PencilIcon, XIcon } from 'lucide-react'
import { useState, useEffect, useRef } from 'react'

import { TopBar } from '@/components/TopBar'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
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

  const { health, isHealthLoading, isHealthError, activeUser, settingsMap, updateSetting, refetchAll } = useSettings()

  // Top right corner Edit Mode State per section
  const [isEditingSection, setIsEditingSection] = useState<Record<string, boolean>>({})

  // Production level defaults
  const DEFAULTS = {
    chatModel: 'qwen3:4b',
    temperature: '0.20',
    maxContextTokens: '6000',
    llmTimeout: '300.0',
    topK: '20',
    finalContext: '4',
    similarityThreshold: '0.35',
    chunkSize: '1000',
    chunkOverlap: '200',
    embeddingModel: 'nomic-embed-text',
    vectorDimensions: '768',
  }

  // Dynamic state for editable AI Settings
  const [chatModel, setChatModel] = useState<string>(DEFAULTS.chatModel)
  const [temperature, setTemperature] = useState<string>(DEFAULTS.temperature)
  const [maxContextTokens, setMaxContextTokens] = useState<string>(DEFAULTS.maxContextTokens)
  const [llmTimeout, setLlmTimeout] = useState<string>(DEFAULTS.llmTimeout)

  // Dynamic state for editable Retrieval Settings
  const [topK, setTopK] = useState<string>(DEFAULTS.topK)
  const [finalContext, setFinalContext] = useState<string>(DEFAULTS.finalContext)
  const [similarityThreshold, setSimilarityThreshold] = useState<string>(DEFAULTS.similarityThreshold)
  const [chunkSize, setChunkSize] = useState<string>(DEFAULTS.chunkSize)
  const [chunkOverlap, setChunkOverlap] = useState<string>(DEFAULTS.chunkOverlap)

  // Dynamic state for editable Embedding Settings
  const [embeddingModel, setEmbeddingModel] = useState<string>(DEFAULTS.embeddingModel)
  const [vectorDimensions, setVectorDimensions] = useState<string>(DEFAULTS.vectorDimensions)

  // Status feedback state
  const [savedSection, setSavedSection] = useState<string | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const isInitializedRef = useRef(false)

  // Populate dynamic settings from backend system-settings map on initial load
  useEffect(() => {
    if (settingsMap.size > 0 && !isInitializedRef.current) {
      isInitializedRef.current = true
      const getVal = (key: string): string | null => {
        const raw = settingsMap.get(key)?.value
        if (raw === undefined || raw === null) return null
        if (typeof raw === 'object' && raw !== null && 'val' in raw) {
          return String((raw as Record<string, unknown>).val)
        }
        return String(raw)
      }

      const cm = getVal('CHAT_MODEL')
      if (cm !== null) setChatModel(cm)
      const temp = getVal('LLM_TEMPERATURE')
      if (temp !== null) setTemperature(temp)
      const mct = getVal('MAX_CONTEXT_TOKENS')
      if (mct !== null) setMaxContextTokens(mct)
      const timeout = getVal('LLM_TIMEOUT')
      if (timeout !== null) setLlmTimeout(timeout)

      const tk = getVal('TOP_K')
      if (tk !== null) setTopK(tk)
      const fc = getVal('FINAL_CONTEXT')
      if (fc !== null) setFinalContext(fc)
      const st = getVal('SIMILARITY_THRESHOLD')
      if (st !== null) setSimilarityThreshold(st)
      const cs = getVal('CHUNK_SIZE')
      if (cs !== null) setChunkSize(cs)
      const co = getVal('CHUNK_OVERLAP')
      if (co !== null) setChunkOverlap(co)

      const em = getVal('EMBEDDING_MODEL')
      if (em !== null) setEmbeddingModel(em)
      const vd = getVal('VECTOR_DIMENSIONS')
      if (vd !== null) setVectorDimensions(vd)
    }
  }, [settingsMap])

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

  const toggleEditSection = (section: string) => {
    setIsEditingSection((prev) => ({ ...prev, [section]: !prev[section] }))
  }

  const handleSaveAISettings = async () => {
    setIsSaving(true)
    try {
      await updateSetting({ key: 'CHAT_MODEL', value: chatModel, description: 'Active Ollama local chat LLM model' })
      await updateSetting({ key: 'LLM_TEMPERATURE', value: Number.parseFloat(temperature) || 0.2, description: 'LLM sampling temperature' })
      await updateSetting({ key: 'MAX_CONTEXT_TOKENS', value: Number.parseInt(maxContextTokens, 10) || 6000, description: 'Max prompt context window tokens' })
      await updateSetting({ key: 'LLM_TIMEOUT', value: Number.parseFloat(llmTimeout) || 300.0, description: 'LLM request timeout seconds' })

      setSavedSection('ai')
      setTimeout(() => setSavedSection(null), 3000)
    } finally {
      setIsSaving(false)
    }
  }

  const handleSaveRetrievalSettings = async () => {
    setIsSaving(true)
    try {
      await updateSetting({ key: 'TOP_K', value: Number.parseInt(topK, 10) || 20, description: 'Top K vector search candidates' })
      await updateSetting({ key: 'FINAL_CONTEXT', value: Number.parseInt(finalContext, 10) || 4, description: 'Final context passages passed to LLM' })
      await updateSetting({ key: 'SIMILARITY_THRESHOLD', value: Number.parseFloat(similarityThreshold) || 0.0, description: 'Vector similarity threshold' })
      await updateSetting({ key: 'CHUNK_SIZE', value: Number.parseInt(chunkSize, 10) || 1000, description: 'Semantic text chunk size in characters' })
      await updateSetting({ key: 'CHUNK_OVERLAP', value: Number.parseInt(chunkOverlap, 10) || 200, description: 'Text chunk overlap in characters' })

      setSavedSection('retrieval')
      setTimeout(() => setSavedSection(null), 3000)
    } finally {
      setIsSaving(false)
    }
  }

  const handleSaveEmbeddingSettings = async () => {
    setIsSaving(true)
    try {
      await updateSetting({ key: 'EMBEDDING_MODEL', value: embeddingModel, description: 'Embedding model specification' })
      await updateSetting({ key: 'VECTOR_DIMENSIONS', value: Number.parseInt(vectorDimensions, 10) || 768, description: 'PostgreSQL pgvector dimensions' })

      setSavedSection('embeddings')
      setTimeout(() => setSavedSection(null), 3000)
    } finally {
      setIsSaving(false)
    }
  }

  const resetAIDefaults = () => {
    setChatModel(DEFAULTS.chatModel)
    setTemperature(DEFAULTS.temperature)
    setMaxContextTokens(DEFAULTS.maxContextTokens)
    setLlmTimeout(DEFAULTS.llmTimeout)
  }

  const resetRetrievalDefaults = () => {
    setTopK(DEFAULTS.topK)
    setFinalContext(DEFAULTS.finalContext)
    setSimilarityThreshold(DEFAULTS.similarityThreshold)
    setChunkSize(DEFAULTS.chunkSize)
    setChunkOverlap(DEFAULTS.chunkOverlap)
  }

  const resetEmbeddingDefaults = () => {
    setEmbeddingModel(DEFAULTS.embeddingModel)
    setVectorDimensions(DEFAULTS.vectorDimensions)
  }

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
                  value={activeUser ? activeUser.email : 'admin@localrag.internal'}
                  copyable={Boolean(activeUser)}
                />
                <InfoRow
                  label="Role"
                  value={<Badge variant="secondary" className="bg-primary/10 text-primary font-semibold">Admin (Full Access)</Badge>}
                />
                <InfoRow
                  label="User ID"
                  value={activeUser ? activeUser.id : 'system-admin-uuid'}
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
                action={
                  <div className="flex items-center gap-2">
                    {isEditingSection['ai'] ? (
                      <>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={resetAIDefaults}
                          title="Reset to production defaults"
                          className="gap-1 text-muted-foreground hover:text-foreground"
                        >
                          <RotateCcwIcon className="size-3.5" />
                          <span>Reset</span>
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          onClick={() => { void handleSaveAISettings() }}
                          disabled={isSaving}
                          className="gap-1.5"
                        >
                          {savedSection === 'ai' ? (
                            <>
                              <CheckIcon className="size-3.5 text-emerald-400" />
                              <span>Saved</span>
                            </>
                          ) : (
                            <>
                              <SaveIcon className="size-3.5" />
                              <span>Save Changes</span>
                            </>
                          )}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => toggleEditSection('ai')}
                          className="gap-1"
                        >
                          <XIcon className="size-3.5" />
                          <span>Done</span>
                        </Button>
                      </>
                    ) : (
                      <Button
                        type="button"
                        variant="default"
                        size="sm"
                        onClick={() => toggleEditSection('ai')}
                        className="gap-1.5"
                      >
                        <PencilIcon className="size-3.5" />
                        <span>Edit Settings</span>
                      </Button>
                    )}
                  </div>
                }
              >
                <InfoRow
                  label="LLM Provider"
                  value={<Badge variant="secondary" className="font-mono">Ollama (Local GPU/CPU)</Badge>}
                />

                {isEditingSection['ai'] ? (
                  <>
                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Chat Model</div>
                        <div className="text-xs text-muted-foreground">Local Ollama model identifier used for RAG generation</div>
                      </div>
                      <Input
                        className="w-full sm:w-48 font-mono text-sm bg-background"
                        value={chatModel}
                        onChange={(e) => setChatModel(e.target.value)}
                      />
                    </div>

                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Temperature</div>
                        <div className="text-xs text-muted-foreground">Controls generation randomness (0.0 = deterministic, 1.0 = creative)</div>
                      </div>
                      <Input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={temperature}
                        onChange={(e) => setTemperature(e.target.value)}
                      />
                    </div>

                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Max Context Tokens</div>
                        <div className="text-xs text-muted-foreground">Maximum token budget allocated for prompt context</div>
                      </div>
                      <Input
                        type="number"
                        step="500"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={maxContextTokens}
                        onChange={(e) => setMaxContextTokens(e.target.value)}
                      />
                    </div>

                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">LLM Timeout (seconds)</div>
                        <div className="text-xs text-muted-foreground">Timeout limit before aborting generation request</div>
                      </div>
                      <Input
                        type="number"
                        step="10"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={llmTimeout}
                        onChange={(e) => setLlmTimeout(e.target.value)}
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <InfoRow label="Chat Model" value={<ReadOnlyValue value={chatModel} />} />
                    <InfoRow label="Temperature" value={<ReadOnlyValue value={temperature} />} />
                    <InfoRow label="Max Context Tokens" value={<ReadOnlyValue value={`${maxContextTokens} tokens`} />} />
                    <InfoRow label="LLM Timeout" value={<ReadOnlyValue value={`${llmTimeout} seconds`} />} />
                  </>
                )}

                <InfoRow
                  label="Streaming Support"
                  value={<Badge variant="outline" className="text-emerald-600 bg-emerald-500/10 border-emerald-500/20">SSE (Server-Sent Events)</Badge>}
                />
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
                action={
                  <div className="flex items-center gap-2">
                    {isEditingSection['retrieval'] ? (
                      <>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={resetRetrievalDefaults}
                          title="Reset to production defaults"
                          className="gap-1 text-muted-foreground hover:text-foreground"
                        >
                          <RotateCcwIcon className="size-3.5" />
                          <span>Reset</span>
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          onClick={() => { void handleSaveRetrievalSettings() }}
                          disabled={isSaving}
                          className="gap-1.5"
                        >
                          {savedSection === 'retrieval' ? (
                            <>
                              <CheckIcon className="size-3.5 text-emerald-400" />
                              <span>Saved</span>
                            </>
                          ) : (
                            <>
                              <SaveIcon className="size-3.5" />
                              <span>Save Changes</span>
                            </>
                          )}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => toggleEditSection('retrieval')}
                          className="gap-1"
                        >
                          <XIcon className="size-3.5" />
                          <span>Done</span>
                        </Button>
                      </>
                    ) : (
                      <Button
                        type="button"
                        variant="default"
                        size="sm"
                        onClick={() => toggleEditSection('retrieval')}
                        className="gap-1.5"
                      >
                        <PencilIcon className="size-3.5" />
                        <span>Edit Settings</span>
                      </Button>
                    )}
                  </div>
                }
              >
                {isEditingSection['retrieval'] ? (
                  <>
                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Top K Candidates</div>
                        <div className="text-xs text-muted-foreground">Initial vector candidate count retrieved before reranking</div>
                      </div>
                      <Input
                        type="number"
                        min="1"
                        max="100"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={topK}
                        onChange={(e) => setTopK(e.target.value)}
                      />
                    </div>

                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Final Context (Passages)</div>
                        <div className="text-xs text-muted-foreground">Number of top-ranked passages passed to LLM prompt context</div>
                      </div>
                      <Input
                        type="number"
                        min="1"
                        max="10"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={finalContext}
                        onChange={(e) => setFinalContext(e.target.value)}
                      />
                    </div>

                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Similarity Threshold</div>
                        <div className="text-xs text-muted-foreground">Minimum similarity score cutoff (0.0 = return best matches)</div>
                      </div>
                      <Input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={similarityThreshold}
                        onChange={(e) => setSimilarityThreshold(e.target.value)}
                      />
                    </div>

                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Chunk Size (characters)</div>
                        <div className="text-xs text-muted-foreground">Target size for recursive text chunking</div>
                      </div>
                      <Input
                        type="number"
                        step="100"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={chunkSize}
                        onChange={(e) => setChunkSize(e.target.value)}
                      />
                    </div>

                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Chunk Overlap (characters)</div>
                        <div className="text-xs text-muted-foreground">Overlapping character margin between consecutive chunks</div>
                      </div>
                      <Input
                        type="number"
                        step="50"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={chunkOverlap}
                        onChange={(e) => setChunkOverlap(e.target.value)}
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <InfoRow label="Top K Candidates" value={<ReadOnlyValue value={`${topK} chunks`} />} />
                    <InfoRow label="Final Context" value={<ReadOnlyValue value={`${finalContext} passages`} />} />
                    <InfoRow label="Similarity Threshold" value={<ReadOnlyValue value={similarityThreshold} />} />
                    <InfoRow label="Chunk Size" value={<ReadOnlyValue value={`${chunkSize} characters`} />} />
                    <InfoRow label="Chunk Overlap" value={<ReadOnlyValue value={`${chunkOverlap} characters`} />} />
                  </>
                )}

                <InfoRow
                  label="Search Strategy"
                  value={<Badge variant="secondary">Hybrid Search (pgvector Cosine + tsvector FTS + RRF Rerank)</Badge>}
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
                action={
                  <div className="flex items-center gap-2">
                    {isEditingSection['embeddings'] ? (
                      <>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={resetEmbeddingDefaults}
                          title="Reset to production defaults"
                          className="gap-1 text-muted-foreground hover:text-foreground"
                        >
                          <RotateCcwIcon className="size-3.5" />
                          <span>Reset</span>
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          onClick={() => { void handleSaveEmbeddingSettings() }}
                          disabled={isSaving}
                          className="gap-1.5"
                        >
                          {savedSection === 'embeddings' ? (
                            <>
                              <CheckIcon className="size-3.5 text-emerald-400" />
                              <span>Saved</span>
                            </>
                          ) : (
                            <>
                              <SaveIcon className="size-3.5" />
                              <span>Save Changes</span>
                            </>
                          )}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => toggleEditSection('embeddings')}
                          className="gap-1"
                        >
                          <XIcon className="size-3.5" />
                          <span>Done</span>
                        </Button>
                      </>
                    ) : (
                      <Button
                        type="button"
                        variant="default"
                        size="sm"
                        onClick={() => toggleEditSection('embeddings')}
                        className="gap-1.5"
                      >
                        <PencilIcon className="size-3.5" />
                        <span>Edit Settings</span>
                      </Button>
                    )}
                  </div>
                }
              >
                {isEditingSection['embeddings'] ? (
                  <>
                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Embedding Model</div>
                        <div className="text-xs text-muted-foreground">Ollama embedding model used for dense vector generation</div>
                      </div>
                      <Input
                        className="w-full sm:w-48 font-mono text-sm bg-background"
                        value={embeddingModel}
                        onChange={(e) => setEmbeddingModel(e.target.value)}
                      />
                    </div>

                    <div className="py-3 border-b border-border/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium text-foreground">Vector Dimensions</div>
                        <div className="text-xs text-muted-foreground">Embedding vector output dimension size</div>
                      </div>
                      <Input
                        type="number"
                        className="w-full sm:w-32 font-mono text-sm bg-background"
                        value={vectorDimensions}
                        onChange={(e) => setVectorDimensions(e.target.value)}
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <InfoRow label="Embedding Model" value={<ReadOnlyValue value={embeddingModel} />} />
                    <InfoRow label="Vector Dimensions" value={<ReadOnlyValue value={vectorDimensions} />} />
                  </>
                )}

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
