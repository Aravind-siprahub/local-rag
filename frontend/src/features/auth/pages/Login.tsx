import React, { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  Eye,
  EyeOff,
  Database,
  Mail,
  Lock,
  ArrowRight,
  Loader2,
  ShieldCheck,
  KeyRound,
  Copy,
  Check,
  CheckCircle2,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { AuthStore } from '@/features/auth/utils/authStore'
import { authApi } from '@/features/auth/api/authApi'
import type { TwoFactorSetupData } from '@/features/auth/types/authTypes'

/* ─── tiny helpers ─────────────────────────────────────────── */
function FieldError({ msg }: { msg?: string }) {
  if (!msg) return null
  return (
    <p className="flex items-center gap-1 text-xs text-red-400 mt-1 animate-in fade-in slide-in-from-top-1 duration-200">
      <span className="inline-block w-1 h-1 rounded-full bg-red-400 shrink-0" />
      {msg}
    </p>
  )
}

/* ─── component ─────────────────────────────────────────────── */
export const Login: React.FC = () => {
  const navigate = useNavigate()

  const [searchParams] = useSearchParams()
  const isJustVerified = searchParams.get('verified') === 'true'
  const isJustReset = searchParams.get('reset') === 'true'
  const [unverifiedNotice, setUnverifiedNotice] = useState<string | null>(null)

  // Primary credentials state
  const [formData, setFormData] = useState({ email: '', password: '', remember: false })
  const [errors, setErrors] = useState<{ email?: string; password?: string; form?: string; totp?: string }>({})
  const [showPassword, setShowPassword] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // 2FA Challenge state
  const [step, setStep] = useState<'credentials' | '2fa_verify' | '2fa_setup'>('credentials')
  const [tempToken, setTempToken] = useState<string>('')
  const [totpCode, setTotpCode] = useState<string>('')
  const [isBackupCodeMode, setIsBackupCodeMode] = useState<boolean>(false)
  const [setupData, setSetupData] = useState<TwoFactorSetupData | null>(null)
  const [copiedSecret, setCopiedSecret] = useState(false)
  const [copiedBackup, setCopiedBackup] = useState(false)

  /* ── field change ── */
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target
    setFormData((prev) => ({ ...prev, [name]: type === 'checkbox' ? checked : value }))
    if (name in errors) setErrors((prev) => ({ ...prev, [name]: undefined, form: undefined }))
  }

  /* ── credentials validation ── */
  const validate = () => {
    const errs: typeof errors = {}
    if (!formData.email.trim()) errs.email = 'Email address is required'
    else if (!/\S+@\S+\.\S+/.test(formData.email)) errs.email = 'Enter a valid email address'
    if (!formData.password) errs.password = 'Password is required'
    setErrors(errs)
    return Object.keys(errs).length === 0
  }

  /* ── step 1: submit credentials ── */
  const handleCredentialsSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    setIsSubmitting(true)
    setErrors({})
    try {
      const result = await authApi.login({
        email: formData.email.trim(),
        password: formData.password,
      })

      if (result.requires2FA) {
        setTempToken(result.tempToken || '')
        if (result.requires2FASetup && result.setupData) {
          setSetupData(result.setupData)
          setStep('2fa_setup')
        } else {
          setStep('2fa_verify')
        }
      } else if (result.auth) {
        AuthStore.setSession(result.auth.accessToken, result.auth.user)
        AuthStore.setRememberedEmail(formData.email, formData.remember)
        window.dispatchEvent(new Event('auth:change'))
        void navigate('/')
      }
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string }; status?: number } }
      const detail = e?.response?.data?.detail
      const status = e?.response?.status

      if (status === 403 && typeof detail === 'string' && detail.toLowerCase().includes('not verified')) {
        setUnverifiedNotice(detail)
        setErrors({})
      } else {
        setUnverifiedNotice(null)
        setErrors({
          form: typeof detail === 'string'
            ? detail.replace(/^Authentication failed:\s*/i, '')
            : 'Invalid email or password. Please try again.',
        })
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  /* ── step 2: submit 2FA verification code ── */
  const handle2FAVerifySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const cleanCode = totpCode.trim()
    if (!cleanCode) {
      setErrors({ totp: isBackupCodeMode ? 'Please enter a recovery code' : 'Please enter your 6-digit authenticator code' })
      return
    }

    setIsSubmitting(true)
    setErrors({})
    try {
      const auth = await authApi.verify2FA({
        tempToken,
        code: cleanCode,
        isBackupCode: isBackupCodeMode,
      })
      AuthStore.setSession(auth.accessToken, auth.user)
      AuthStore.setRememberedEmail(formData.email, formData.remember)
      window.dispatchEvent(new Event('auth:change'))
      void navigate('/')
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string }; status?: number } }
      const detail = e?.response?.data?.detail
      setErrors({
        totp: typeof detail === 'string'
          ? detail
          : 'Invalid verification code. Please try again.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  /* ── step 3: submit first 2FA code during initial setup ── */
  const handle2FAEnableSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const cleanCode = totpCode.trim()
    if (!cleanCode) {
      setErrors({ totp: 'Please enter the 6-digit code shown in your authenticator app' })
      return
    }

    setIsSubmitting(true)
    setErrors({})
    try {
      const auth = await authApi.enable2FA({
        tempToken: tempToken || setupData?.tempToken,
        code: cleanCode,
      })
      AuthStore.setSession(auth.accessToken, auth.user)
      AuthStore.setRememberedEmail(formData.email, formData.remember)
      window.dispatchEvent(new Event('auth:change'))
      void navigate('/')
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string }; status?: number } }
      const detail = e?.response?.data?.detail
      setErrors({
        totp: typeof detail === 'string'
          ? detail
          : 'Invalid authenticator code. Please check your app and try again.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const copyToClipboard = (text: string, type: 'secret' | 'backup') => {
    navigator.clipboard.writeText(text)
    if (type === 'secret') {
      setCopiedSecret(true)
      setTimeout(() => setCopiedSecret(false), 2000)
    } else {
      setCopiedBackup(true)
      setTimeout(() => setCopiedBackup(false), 2000)
    }
  }

  return (
    <div className="w-full">
      {/* ── Logo ── */}
      <div className="flex items-center gap-3 mb-8">
        <div className="w-10 h-10 rounded-xl bg-linear-to-tr from-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
          <Database className="w-5 h-5 text-white" />
        </div>
        <span className="text-xl font-bold tracking-tight text-white font-display">
          Talk to My Data
        </span>
      </div>

      {/* ── Heading ── */}
      <div className="mb-8">
        <h1 className="text-3xl font-extrabold text-foreground tracking-tight font-display leading-tight">
          {step === 'credentials' && 'Welcome back'}
          {step === '2fa_verify' && 'Two-Factor Authentication'}
          {step === '2fa_setup' && 'Set Up Authenticator App'}
        </h1>
        <p className="mt-1.5 text-sm text-muted-foreground">
          {step === 'credentials' && 'Sign in to continue to your knowledge base'}
          {step === '2fa_verify' && (isBackupCodeMode ? 'Enter one of your emergency recovery codes' : 'Enter the 6-digit code from Google/Microsoft Authenticator')}
          {step === '2fa_setup' && 'Scan the QR code with your authenticator app to secure your account'}
        </p>
      </div>

      {/* ── Theme Card ── */}
      <div className="relative rounded-2xl border border-border bg-card shadow-lg p-8 space-y-5">
        <div className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-primary/50 to-transparent rounded-t-2xl" />

        {/* Email verification success notice */}
        {isJustVerified && (
          <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm flex items-center gap-2.5">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>Email verified successfully! Please sign in with your credentials.</span>
          </div>
        )}

        {/* Password reset success notice */}
        {isJustReset && (
          <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm flex items-center gap-2.5">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>Password reset successfully! Please sign in with your new password.</span>
          </div>
        )}

        {/* Unverified account notice with direct action */}
        {unverifiedNotice && (
          <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-sm flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
              <span>{unverifiedNotice}</span>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                sessionStorage.setItem('pendingVerificationEmail', formData.email.trim().toLowerCase())
                navigate('/signup')
              }}
              className="self-start text-xs border-amber-500/40 text-amber-200 hover:bg-amber-500/20"
            >
              Verify Email Now
              <ArrowRight className="w-3 h-3 ml-1" />
            </Button>
          </div>
        )}

        {/* ─── STEP 1: CREDENTIALS ─── */}
        {step === 'credentials' && (
          <form onSubmit={handleCredentialsSubmit} className="space-y-5" noValidate>
            {/* Email */}
            <div className="space-y-1.5">
              <Label htmlFor="email" className="text-sm font-medium text-muted-foreground">
                Email address
              </Label>
              <div className="relative group">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
                <Input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  placeholder="name@example.com"
                  value={formData.email}
                  onChange={handleChange}
                  className={`pl-10 h-10 bg-muted/10 border-border/80 text-foreground placeholder:text-muted-foreground/45
                    focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary/30
                    hover:border-border transition-colors
                    ${errors.email ? 'border-destructive/65 focus-visible:border-destructive focus-visible:ring-destructive/30' : ''}`}
                  disabled={isSubmitting}
                />
              </div>
              <FieldError msg={errors.email} />
            </div>

            {/* Password */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label htmlFor="password" className="text-sm font-medium text-muted-foreground">
                  Password
                </Label>
                <Link
                  to="/forgot-password"
                  className="text-xs text-primary hover:underline font-medium transition-colors"
                >
                  Forgot password?
                </Link>
              </div>
              <div className="relative group">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
                <Input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  placeholder="••••••••"
                  value={formData.password}
                  onChange={handleChange}
                  className={`pl-10 pr-10 h-10 bg-muted/10 border-border/80 text-foreground placeholder:text-muted-foreground/45
                    focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary/30
                    hover:border-border transition-colors
                    ${errors.password ? 'border-destructive/65 focus-visible:border-destructive focus-visible:ring-destructive/30' : ''}`}
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground/60 hover:text-foreground transition-colors"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <FieldError msg={errors.password} />
            </div>

            {/* General form error */}
            {errors.form && (
              <div className="p-3.5 rounded-xl bg-destructive/10 border border-destructive/25 text-destructive text-sm flex items-start gap-2.5">
                <span className="w-1.5 h-1.5 rounded-full bg-destructive mt-1.5 shrink-0" />
                <span>{errors.form}</span>
              </div>
            )}

            {/* Submit */}
            <Button
              type="submit"
              disabled={isSubmitting}
              className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl transition-all shadow-md shadow-primary/20"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Verifying credentials...
                </>
              ) : (
                <>
                  Sign in
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          </form>
        )}

        {/* ─── STEP 2: 2FA VERIFICATION ─── */}
        {step === '2fa_verify' && (
          <form onSubmit={handle2FAVerifySubmit} className="space-y-5" noValidate>
            <div className="flex items-center justify-center p-3 rounded-xl bg-primary/10 border border-primary/20 text-primary mb-2">
              <ShieldCheck className="w-8 h-8 mr-2 text-indigo-400" />
              <span className="text-sm font-medium text-foreground">
                {isBackupCodeMode ? 'Recovery Code Verification' : 'Authenticator App Verification'}
              </span>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="totpCode" className="text-sm font-medium text-muted-foreground">
                {isBackupCodeMode ? 'Backup / Recovery Code' : '6-Digit Authenticator Code'}
              </Label>
              <div className="relative group">
                <KeyRound className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
                <Input
                  id="totpCode"
                  name="totpCode"
                  type="text"
                  autoFocus
                  autoComplete="one-time-code"
                  placeholder={isBackupCodeMode ? 'e.g. A1B2-C3D4' : '000000'}
                  value={totpCode}
                  maxLength={isBackupCodeMode ? 12 : 6}
                  onChange={(e) => {
                    setTotpCode(e.target.value)
                    if (errors.totp) setErrors({})
                  }}
                  className={`pl-10 h-12 text-center tracking-widest text-lg font-mono bg-muted/10 border-border/80 text-foreground
                    focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary/30
                    ${errors.totp ? 'border-destructive/65 focus-visible:border-destructive' : ''}`}
                  disabled={isSubmitting}
                />
              </div>
              <FieldError msg={errors.totp} />
            </div>

            <div className="flex items-center justify-between text-xs pt-1">
              <button
                type="button"
                onClick={() => {
                  setIsBackupCodeMode((v) => !v)
                  setTotpCode('')
                  setErrors({})
                }}
                className="text-primary hover:underline font-medium"
              >
                {isBackupCodeMode ? '← Use 6-Digit App Code' : 'Lost device? Use a recovery code'}
              </button>

              <button
                type="button"
                onClick={() => {
                  setStep('credentials')
                  setTotpCode('')
                  setErrors({})
                }}
                className="text-muted-foreground hover:text-foreground"
              >
                Back to login
              </button>
            </div>

            <Button
              type="submit"
              disabled={isSubmitting || !totpCode.trim()}
              className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl transition-all shadow-md shadow-primary/20"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Verifying 2FA...
                </>
              ) : (
                <>
                  Verify & Sign in
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          </form>
        )}

        {/* ─── STEP 3: 2FA SETUP (If user has not completed setup) ─── */}
        {step === '2fa_setup' && setupData && (
          <form onSubmit={handle2FAEnableSubmit} className="space-y-5" noValidate>
            <div className="text-center space-y-3">
              {setupData.qrCodeDataUrl && (
                <div className="inline-block p-3 bg-white rounded-2xl shadow-md">
                  <img
                    src={setupData.qrCodeDataUrl}
                    alt="2FA QR Code"
                    className="w-44 h-44 mx-auto"
                  />
                </div>
              )}

              <div className="text-left bg-muted/20 p-3 rounded-xl border border-border space-y-1.5">
                <div className="flex items-center justify-between text-xs text-muted-foreground font-medium">
                  <span>Manual Setup Key:</span>
                  <button
                    type="button"
                    onClick={() => copyToClipboard(setupData.totpSecret, 'secret')}
                    className="flex items-center gap-1 text-primary hover:underline"
                  >
                    {copiedSecret ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    {copiedSecret ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <div className="font-mono text-xs text-foreground tracking-wider break-all select-all font-semibold">
                  {setupData.totpSecret}
                </div>
              </div>
            </div>

            {setupData.backupCodes && setupData.backupCodes.length > 0 && (
              <div className="bg-muted/15 p-3.5 rounded-xl border border-border space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-foreground">
                  <span>Emergency Recovery Codes</span>
                  <button
                    type="button"
                    onClick={() => copyToClipboard(setupData.backupCodes.join('\n'), 'backup')}
                    className="flex items-center gap-1 text-primary hover:underline text-xs"
                  >
                    {copiedBackup ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    {copiedBackup ? 'Copied' : 'Copy all'}
                  </button>
                </div>
                <p className="text-[11px] text-muted-foreground">
                  Save these codes now. Each code can only be used once if you lose your phone.
                </p>
                <div className="grid grid-cols-2 gap-1.5 font-mono text-xs bg-background/50 p-2.5 rounded-lg border border-border/50 text-foreground">
                  {setupData.backupCodes.map((code, i) => (
                    <div key={i} className="text-center font-medium">
                      {code}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="space-y-1.5">
              <Label htmlFor="setupCode" className="text-sm font-medium text-muted-foreground">
                Enter 6-Digit Code from App to Confirm
              </Label>
              <Input
                id="setupCode"
                name="setupCode"
                type="text"
                autoFocus
                autoComplete="one-time-code"
                placeholder="000000"
                value={totpCode}
                maxLength={6}
                onChange={(e) => {
                  setTotpCode(e.target.value)
                  if (errors.totp) setErrors({})
                }}
                className="h-11 text-center tracking-widest text-lg font-mono bg-muted/10 border-border/80 text-foreground focus-visible:border-primary"
                disabled={isSubmitting}
              />
              <FieldError msg={errors.totp} />
            </div>

            <Button
              type="submit"
              disabled={isSubmitting || totpCode.trim().length !== 6}
              className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl transition-all shadow-md shadow-primary/20"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Activating 2FA...
                </>
              ) : (
                <>
                  Activate 2FA & Continue
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          </form>
        )}
      </div>

      {/* ── Sign up link ── */}
      {step === 'credentials' && (
        <p className="mt-6 text-center text-sm text-muted-foreground">
          Don&apos;t have an account?{' '}
          <Link
            to="/signup"
            className="font-semibold text-primary hover:text-primary/80 transition-colors"
          >
            Create account
          </Link>
        </p>
      )}
    </div>
  )
}
