export type ThemeMode = 'dark' | 'light' | 'system'

export interface AppSettings {
  userId: string | null
  ollamaBaseUrl: string
  chatModel: string
  embeddingModel: string
  theme: ThemeMode
}

const STORAGE_KEY = 'local-rag-settings'

const defaultSettings: AppSettings = {
  userId: null,
  ollamaBaseUrl: 'http://localhost:11434',
  chatModel: 'qwen3:8b',
  embeddingModel: 'nomic-embed-text',
  theme: 'system',
}

function readSettings(): AppSettings {
  if (typeof window === 'undefined') {
    return defaultSettings
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return defaultSettings
    }

    const parsed = JSON.parse(raw) as Partial<AppSettings>
    return { ...defaultSettings, ...parsed }
  } catch {
    return defaultSettings
  }
}

function writeSettings(settings: AppSettings): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
}

let cachedSettings = readSettings()
const listeners = new Set<() => void>()

export const settingsStore = {
  get(): AppSettings {
    return cachedSettings
  },

  subscribe(listener: () => void): () => void {
    listeners.add(listener)
    return () => listeners.delete(listener)
  },

  set(partial: Partial<AppSettings>): AppSettings {
    cachedSettings = { ...cachedSettings, ...partial }
    writeSettings(cachedSettings)
    listeners.forEach((listener) => listener())
    return cachedSettings
  },

  reset(): AppSettings {
    cachedSettings = defaultSettings
    writeSettings(cachedSettings)
    listeners.forEach((listener) => listener())
    return cachedSettings
  },
}

export function applyTheme(theme: ThemeMode): void {
  if (typeof window === 'undefined') return
  const root = document.documentElement
  let isDark = theme === 'dark'
  if (theme === 'system') {
    isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  }
  root.classList.toggle('light', !isDark)
  root.classList.toggle('dark', isDark)
  window.localStorage.setItem('theme', theme)
}

applyTheme(cachedSettings.theme)
