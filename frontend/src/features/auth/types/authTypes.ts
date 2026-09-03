/**
 * Authentication Data Models & API Interfaces for "Talk to My Data"
 */

export interface User {
  id: string
  email: string
  fullName: string
  avatarUrl?: string
  role?: 'user' | 'admin'
  createdAt?: string
}

export interface LoginCredentials {
  email: string
  password: string
  rememberMe?: boolean
}

export interface SignupCredentials {
  fullName: string
  email: string
  password: string
  confirmPassword: string
  acceptTerms: boolean
}

export interface AuthResponse {
  accessToken: string
  refreshToken?: string
  user: User
  expiresIn?: number
}

export interface ApiErrorResponse {
  message: string
  code?: string
  errors?: Record<string, string[]>
}

export interface TwoFactorSetupData {
  totpSecret: string
  provisioningUri: string
  qrCodeDataUrl: string
  backupCodes: string[]
  tempToken?: string
  message?: string
}

export interface TwoFactorVerifyPayload {
  tempToken?: string
  code: string
  isBackupCode?: boolean
}

export interface LoginResult {
  requires2FA: boolean
  requires2FASetup?: boolean
  tempToken?: string
  auth?: AuthResponse
  setupData?: TwoFactorSetupData
}

export interface RegisterResult {
  status: string
  email: string
  message: string
}

export interface VerifyEmailPayload {
  email: string
  code: string
}

export interface ResendVerificationPayload {
  email: string
}

export interface VerifyEmailResponse {
  status: string
  message: string
  access_token?: string
  token_type?: string
  user?: User
}

export interface ForgotPasswordPayload {
  email: string
}

export interface ForgotPasswordResponse {
  status: string
  message: string
}

export interface ResetPasswordPayload {
  email: string
  code: string
  new_password: string
}

export interface ResetPasswordResponse {
  status: string
  message: string
}

