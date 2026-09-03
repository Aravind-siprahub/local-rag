export const ROUTES = {
  dashboard: '/',
  login: '/login',
  signup: '/signup',
  forgotPassword: '/forgot-password',
  documents: '/documents',
  upload: '/upload',
  chat: '/chat',
  settings: '/settings',
} as const

export type AppRoute = (typeof ROUTES)[keyof typeof ROUTES]
