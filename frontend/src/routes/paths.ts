export const ROUTES = {
  dashboard: '/',
  documents: '/documents',
  upload: '/upload',
  chat: '/chat',
  settings: '/settings',
} as const

export type AppRoute = (typeof ROUTES)[keyof typeof ROUTES]
