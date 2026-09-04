import apiClient from '@/lib/axios'
import type {
  AuthResponse,
  ForgotPasswordPayload,
  ForgotPasswordResponse,
  LoginCredentials,
  LoginResult,
  RegisterResult,
  ResendVerificationPayload,
  ResetPasswordPayload,
  ResetPasswordResponse,
  SignupCredentials,
  TwoFactorSetupData,
  TwoFactorVerifyPayload,
  User,
  VerifyEmailPayload,
  VerifyEmailResponse,
} from '../types/authTypes'

/**
 * Authentication API Service with Password + TOTP 2FA support.
 */
export const authApi = {
  /**
   * User Login API Call
   * POST /auth/login
   */
  async login(credentials: LoginCredentials): Promise<LoginResult> {
    const response = await apiClient.post<{
      requires_2fa?: boolean
      requires_2fa_setup?: boolean
      temp_token?: string
      access_token?: string
      token_type?: string
      user?: { id: string; email: string; full_name?: string; role?: string; created_at?: string; is_2fa_enabled?: boolean }
      setup_data?: {
        totp_secret: string
        provisioning_uri: string
        qr_code_data_url: string
        backup_codes: string[]
        temp_token?: string
      }
    }>('/auth/login', {
      email: credentials.email,
      password: credentials.password,
    })

    const raw = response.data

    if (raw.requires_2fa) {
      let setupData: TwoFactorSetupData | undefined
      if (raw.setup_data) {
        setupData = {
          totpSecret: raw.setup_data.totp_secret,
          provisioningUri: raw.setup_data.provisioning_uri,
          qrCodeDataUrl: raw.setup_data.qr_code_data_url,
          backupCodes: raw.setup_data.backup_codes || [],
          tempToken: raw.setup_data.temp_token,
        }
      }
      return {
        requires2FA: true,
        requires2FASetup: !!raw.requires_2fa_setup,
        tempToken: raw.temp_token,
        setupData,
      }
    }

    if (raw.access_token && raw.user) {
      return {
        requires2FA: false,
        auth: {
          accessToken: raw.access_token,
          user: {
            id: raw.user.id,
            email: raw.user.email,
            fullName: raw.user.full_name ?? raw.user.email.split('@')[0],
            role: (raw.user.role as User['role']) ?? 'user',
            createdAt: raw.user.created_at,
          },
        },
      }
    }

    throw new Error('Unexpected login response from server.')
  },

  /**
   * Verify TOTP code or backup recovery code during login
   * POST /auth/verify-2fa
   */
  async verify2FA(payload: TwoFactorVerifyPayload): Promise<AuthResponse> {
    const response = await apiClient.post<{
      access_token: string
      token_type: string
      user: { id: string; email: string; full_name?: string; role?: string; created_at?: string }
    }>('/auth/verify-2fa', {
      temp_token: payload.tempToken,
      code: payload.code,
      is_backup_code: payload.isBackupCode ?? false,
    })

    const raw = response.data
    return {
      accessToken: raw.access_token,
      user: {
        id: raw.user.id,
        email: raw.user.email,
        fullName: raw.user.full_name ?? raw.user.email.split('@')[0],
        role: (raw.user.role as User['role']) ?? 'user',
        createdAt: raw.user.created_at,
      },
    }
  },

  /**
   * User Registration / Sign Up API Call
   * POST /auth/register
   * Registers account and triggers email verification OTP dispatch.
   */
  async register(credentials: SignupCredentials): Promise<RegisterResult> {
    const response = await apiClient.post<RegisterResult>('/auth/register', {
      full_name: credentials.fullName,
      email: credentials.email,
      password: credentials.password,
    })
    return response.data
  },

  /**
   * Verify Email Address with 6-digit OTP
   * POST /auth/verify-email
   */
  async verifyEmail(payload: VerifyEmailPayload): Promise<VerifyEmailResponse> {
    const response = await apiClient.post<VerifyEmailResponse>('/auth/verify-email', {
      email: payload.email,
      code: payload.code,
    })
    return response.data
  },

  /**
   * Resend Verification Code
   * POST /auth/resend-verification
   */
  async resendVerification(payload: ResendVerificationPayload): Promise<{ status: string; message: string }> {
    const response = await apiClient.post<{ status: string; message: string }>('/auth/resend-verification', {
      email: payload.email,
    })
    return response.data
  },

  /**
   * Request Password Reset Code
   * POST /auth/forgot-password
   */
  async forgotPassword(payload: ForgotPasswordPayload): Promise<ForgotPasswordResponse> {
    const response = await apiClient.post<ForgotPasswordResponse>('/auth/forgot-password', {
      email: payload.email,
    })
    return response.data
  },

  /**
   * Reset Password with OTP Code
   * POST /auth/reset-password
   */
  async resetPassword(payload: ResetPasswordPayload): Promise<ResetPasswordResponse> {
    const response = await apiClient.post<ResetPasswordResponse>('/auth/reset-password', {
      email: payload.email,
      code: payload.code,
      new_password: payload.new_password,
    })
    return response.data
  },

  /**
   * Enable 2FA after scanning QR during registration / onboarding
   * POST /auth/2fa/enable
   */
  async enable2FA(payload: TwoFactorVerifyPayload): Promise<AuthResponse> {
    const response = await apiClient.post<{
      access_token: string
      token_type: string
      user: { id: string; email: string; full_name?: string; role?: string; created_at?: string }
    }>('/auth/2fa/enable', {
      temp_token: payload.tempToken,
      code: payload.code,
    })

    const raw = response.data
    return {
      accessToken: raw.access_token,
      user: {
        id: raw.user.id,
        email: raw.user.email,
        fullName: raw.user.full_name ?? raw.user.email.split('@')[0],
        role: (raw.user.role as User['role']) ?? 'user',
        createdAt: raw.user.created_at,
      },
    }
  },

  /**
   * Fetch Logged-in User Profile
   * GET /auth/me
   */
  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>('/auth/me')
    return response.data
  },

  /**
   * User Logout Call
   * POST /auth/logout
   */
  async logout(): Promise<void> {
    try {
      await apiClient.post('/auth/logout')
    } catch {
      // Ignore network errors on logout
    }
  },
}
