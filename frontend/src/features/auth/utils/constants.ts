/**
 * Constants & Configuration for Authentication
 */

export const AUTH_KEYS = {
  ACCESS_TOKEN: 'talk_to_my_data_access_token',
  REFRESH_TOKEN: 'talk_to_my_data_refresh_token',
  USER_DATA: 'talk_to_my_data_user',
  REMEMBERED_EMAIL: 'talk_to_my_data_remembered_email',
} as const

export const SOCIAL_AUTH_URLS = {
  GOOGLE: '/api/auth/google',
  MICROSOFT: '/api/auth/microsoft',
} as const

export const PASSWORD_REQUIREMENTS = {
  MIN_LENGTH: 8,
  REQUIRE_UPPERCASE: true,
  REQUIRE_LOWERCASE: true,
  REQUIRE_NUMBER: true,
  REQUIRE_SPECIAL: true,
} as const
