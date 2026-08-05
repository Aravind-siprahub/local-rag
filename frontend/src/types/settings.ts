export interface SystemSetting {
  key: string
  value: unknown
  description: string | null
  created_at: string
  updated_at: string
}

export type SystemSettingListResponse = import('./api').PaginatedResponse<SystemSetting>

export interface SystemSettingUpsertPayload {
  value: unknown
  description?: string
  updated_by?: string
}

export type SettingsSectionId =
  | 'general'
  | 'ai'
  | 'retrieval'
  | 'embeddings'
  | 'appearance'
  | 'system'
  | 'about'

export interface SettingsNavItem {
  id: SettingsSectionId
  label: string
  description: string
}
