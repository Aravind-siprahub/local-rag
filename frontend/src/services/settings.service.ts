import { apiClient } from '@/api/client'
import type {
  PaginationParams,
  SystemSetting,
  SystemSettingListResponse,
  SystemSettingUpsertPayload,
} from '@/types'

export async function listSystemSettings(
  params: PaginationParams = {},
): Promise<SystemSettingListResponse> {
  const { data } = await apiClient.get<SystemSettingListResponse>('/system-settings', { params })
  return data
}

export async function getSystemSetting(key: string): Promise<SystemSetting> {
  const { data } = await apiClient.get<SystemSetting>(`/system-settings/${key}`)
  return data
}

export async function upsertSystemSetting(
  key: string,
  payload: SystemSettingUpsertPayload,
): Promise<SystemSetting> {
  const { data } = await apiClient.put<SystemSetting>(`/system-settings/${key}`, payload)
  return data
}

export async function deleteSystemSetting(key: string): Promise<void> {
  await apiClient.delete(`/system-settings/${key}`)
}
