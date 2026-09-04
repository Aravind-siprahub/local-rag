import React, { useState, useMemo, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Eye,
  EyeOff,
  Database,
  Mail,
  Lock,
  User,
  ArrowRight,
  Loader2,
  CheckCircle2,
  XCircle,
  KeyRound,
  RotateCcw,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { AuthStore } from '@/features/auth/utils/authStore'
import { authApi } from '@/features/auth/api/authApi'
import type { User as AuthUser } from '@/features/auth/types/authTypes'

/* ─── helpers ───────────────────────────────────────────────── */
function FieldError({ msg }: { msg?: string }) {
  if (!msg) return null
  return (
    <p className="flex items-center gap-1 text-xs text-red-400 mt-1 animate-in fade-in slide-in-from-top-1 duration-200">
      <span className="inline-block w-1 h-1 rounded-full bg-red-400 shrink-0" />
      {msg}
    </p>
  )
}

function formatCooldown(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s < 10 ? '0' : ''}${s}`
}

/* ── password strength ── */
interface StrengthRule {
  label: string
  test: (pw: string) => boolean
}
const RULES: StrengthRule[] = [
  { label: 'At least 8 characters', test: (pw) => pw.length >= 8 },
  { label: 'Uppercase letter', test: (pw) => /[A-Z]/.test(pw) },
  { label: 'Lowercase letter', test: (pw) => /[a-z]/.test(pw) },
  { label: 'Number', test: (pw) => /[0-9]/.test(pw) },
  { label: 'Special character (!@#$%^&* etc.)', test: (pw) => /[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;/`~]/.test(pw) },
]

const STRENGTH_LABELS = ['', 'Very Weak', 'Weak', 'Fair', 'Good', 'Strong'] as const
const STRENGTH_COLORS = ['', 'bg-red-600', 'bg-red-500', 'bg-amber-500', 'bg-yellow-400', 'bg-emerald-500'] as const
const STRENGTH_TEXT = ['', 'text-red-500', 'text-red-400', 'text-amber-400', 'text-yellow-400', 'text-emerald-400'] as const

function PasswordStrength({ password }: { password: string }) {
  const strength = useMemo(() => RULES.filter((r) => r.test(password)).length, [password])
  if (!password) return null
  return (
    <div className="mt-2.5 space-y-2 animate-in fade-in slide-in-from-top-1 duration-200">
      <div className="flex gap-1">
        {RULES.map((_, i) => (
          <div
            key={i}
            className={`h-1 flex-1 rounded-full transition-all duration-300 ${
              i < strength ? STRENGTH_COLORS[strength] : 'bg-muted'
            }`}
          />
        ))}
      </div>
      <p className={`text-xs font-medium ${STRENGTH_TEXT[strength]}`}>
        {STRENGTH_LABELS[strength]}
      </p>
      <ul className="space-y-1">
        {RULES.map((rule) => {
          const ok = rule.test(password)
          return (
            <li key={rule.label} className="flex items-center gap-1.5 text-[11px]">
              {ok ? (
                <CheckCircle2 className="w-3 h-3 text-emerald-500 shrink-0" />
              ) : (
                <XCircle className="w-3 h-3 text-muted-foreground/30 shrink-0" />
              )}
              <span className={ok ? 'text-foreground/90' : 'text-muted-foreground/60'}>{rule.label}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

/* ── named input field ── */
interface FieldProps {
  id: string
  name: string
  label: string
  type?: string
  placeholder: string
  icon: React.ElementType
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  disabled?: boolean
  rightSlot?: React.ReactNode
  error?: string
  autoComplete?: string
}

function Field({
  id,
  name,
  label,
  type = 'text',
  placeholder,
  icon: Icon,
  value,
  onChange,
  disabled,
  rightSlot,
  error,
  autoComplete,
}: FieldProps) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="text-sm font-medium text-muted-foreground">
        {label}
      </Label>
      <div className="relative group">
        <Icon className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
        <Input
          id={id}
          name={name}
          type={type}
          autoComplete={autoComplete}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          disabled={disabled}
          className={`pl-10 h-10 ${rightSlot ? 'pr-10' : ''} bg-muted/10 border-border/80 text-foreground placeholder:text-muted-foreground/45
            focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary/30
            hover:border-border transition-colors
            ${error ? 'border-destructive/65 focus-visible:border-destructive focus-visible:ring-destructive/30' : ''}`}
        />
        {rightSlot}
      </div>
      <FieldError msg={error} />
    </div>
  )
}

/* ─── component ─────────────────────────────────────────────── */
export const Signup: React.FC = () => {
  const navigate = useNavigate()

  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
    acceptTerms: true,
  })

  const [errors, setErrors] = useState<{
    fullName?: string
    email?: string
    password?: string
    confirmPassword?: string
    acceptTerms?: string
    form?: string
    otp?: string
  }>({})

  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // 3-Stage Auth State: 'register' -> 'verify_email' -> 'verified'
  const [step, setStep] = useState<'register' | 'verify_email' | 'verified'>('register')
  const [pendingEmail, setPendingEmail] = useState<string>('')
  const [otpCode, setOtpCode] = useState<string>('')
  const [resendCooldown, setResendCooldown] = useState<number>(0)
  const [isResending, setIsResending] = useState<boolean>(false)
  const [resendFeedback, setResendFeedback] = useState<string | null>(null)

  // Restore verification state on page reload (Requirement 15)
  useEffect(() => {
    const stored = sessionStorage.getItem('pendingVerificationEmail')
    if (stored) {
      setPendingEmail(stored)
      setStep('verify_email')
    }
  }, [])

  // Cooldown countdown timer
  useEffect(() => {
    if (resendCooldown <= 0) return
    const timer = setInterval(() => {
      setResendCooldown((prev) => Math.max(0, prev - 1))
    }, 1000)
    return () => clearInterval(timer)
  }, [resendCooldown])

  /* ── field change ── */
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target
    setFormData((prev) => ({ ...prev, [name]: type === 'checkbox' ? checked : value }))
    if (name in errors) setErrors((prev) => ({ ...prev, [name]: undefined, form: undefined }))
  }

  /* ── validation ── */
  const validate = () => {
    const errs: typeof errors = {}
    if (!formData.fullName.trim()) errs.fullName = 'Full name is required'
    if (!formData.email.trim()) errs.email = 'Email address is required'
    else if (!/\S+@\S+\.\S+/.test(formData.email)) errs.email = 'Enter a valid email address'
    if (!formData.password) errs.password = 'Password is required'
    else if (formData.password.length < 8) errs.password = 'Password must be at least 8 characters'
    if (!formData.confirmPassword) errs.confirmPassword = 'Please confirm your password'
    else if (formData.password !== formData.confirmPassword)
      errs.confirmPassword = 'Passwords do not match'
    if (!formData.acceptTerms) errs.acceptTerms = 'You must accept the terms to continue'
    setErrors(errs)
    return Object.keys(errs).length === 0
  }

  /* ── submit step 1: register account & dispatch OTP ── */
  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    setIsSubmitting(true)
    setErrors({})
    setResendFeedback(null)

    const cleanEmail = formData.email.trim().toLowerCase()

    try {
      await authApi.register({
        fullName: formData.fullName.trim(),
        email: cleanEmail,
        password: formData.password,
        confirmPassword: formData.confirmPassword,
        acceptTerms: formData.acceptTerms,
      })

      setPendingEmail(cleanEmail)
      sessionStorage.setItem('pendingVerificationEmail', cleanEmail)
      setResendCooldown(300)
      setStep('verify_email')
    } catch (err: unknown) {
      const e = err as { response?: { status?: number; data?: { detail?: string } } }
      const status = e?.response?.status
      const detail = e?.response?.data?.detail

      if (status === 409) {
        setErrors({ form: 'Email already registered. Please login.' })
      } else {
        setErrors({
          form: typeof detail === 'string'
            ? detail
            : 'Account registration failed. Please try again.',
        })
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  /* ── submit step 2: verify 6-digit email OTP ── */
  const handleVerifyOtpSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const cleanCode = otpCode.trim()
    if (!cleanCode || cleanCode.length < 4) {
      setErrors({ otp: 'Please enter the 6-digit verification code from your email' })
      return
    }

    setIsSubmitting(true)
    setErrors({})
    setResendFeedback(null)

    try {
      const res = await authApi.verifyEmail({
        email: pendingEmail,
        code: cleanCode,
      })

      sessionStorage.removeItem('pendingVerificationEmail')

      if (res.access_token && res.user) {
        const normalized: AuthUser = {
          id: res.user.id,
          email: res.user.email,
          fullName: res.user.fullName || (res.user as unknown as { full_name?: string }).full_name || res.user.email.split('@')[0],
          role: (res.user.role as AuthUser['role']) || 'member',
          createdAt: res.user.createdAt,
        }
        AuthStore.setSession(res.access_token, normalized)
        window.dispatchEvent(new Event('auth:change'))
        void navigate('/')
      } else {
        setStep('verified')
      }
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string }; status?: number } }
      const detail = e?.response?.data?.detail
      setErrors({
        otp: typeof detail === 'string'
          ? detail
          : 'Invalid or expired verification code. Please check and try again.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  /* ── resend OTP handler ── */
  const handleResendOtp = async () => {
    if (resendCooldown > 0 || isResending || !pendingEmail) return
    setIsResending(true)
    setErrors({})
    setResendFeedback(null)

    try {
      const res = await authApi.resendVerification({ email: pendingEmail })
      setResendCooldown(300)
      setResendFeedback(res.message || 'New verification code sent!')
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string }; status?: number } }
      const detail = e?.response?.data?.detail
      setErrors({
        otp: typeof detail === 'string'
          ? detail
          : 'Could not resend verification code. Please wait a moment and try again.',
      })
    } finally {
      setIsResending(false)
    }
  }

  /* ── reset back to register with different email ── */
  const handleEditEmail = () => {
    sessionStorage.removeItem('pendingVerificationEmail')
    setStep('register')
    setOtpCode('')
    setErrors({})
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
          {step === 'register' && 'Create your account'}
          {step === 'verify_email' && 'Check your email'}
          {step === 'verified' && 'Email Verified!'}
        </h1>
        <p className="mt-1.5 text-sm text-muted-foreground">
          {step === 'register' && 'Start querying your documents with local AI in minutes'}
          {step === 'verify_email' && (
            <span>
              We sent a 6-digit verification code to{' '}
              <strong className="text-foreground font-semibold">{pendingEmail}</strong>
            </span>
          )}
          {step === 'verified' && 'Your account is ready. Please log in to continue.'}
        </p>
      </div>

      {/* ── Theme Card ── */}
      <div className="relative rounded-2xl border border-border bg-card shadow-lg p-8 space-y-5">
        <div className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-primary/50 to-transparent rounded-t-2xl" />

        {/* ─── STEP 1: REGISTRATION FORM ─── */}
        {step === 'register' && (
          <form onSubmit={handleRegisterSubmit} className="space-y-4" noValidate>
            <Field
              id="fullName"
              name="fullName"
              label="Full name"
              autoComplete="name"
              placeholder="Aravind S."
              icon={User}
              value={formData.fullName}
              onChange={handleChange}
              disabled={isSubmitting}
              error={errors.fullName}
            />

            <Field
              id="email"
              name="email"
              label="Email address"
              type="email"
              autoComplete="email"
              placeholder="name@example.com"
              icon={Mail}
              value={formData.email}
              onChange={handleChange}
              disabled={isSubmitting}
              error={errors.email}
            />

            <Field
              id="password"
              name="password"
              label="Password"
              type={showPassword ? 'text' : 'password'}
              autoComplete="new-password"
              placeholder="••••••••"
              icon={Lock}
              value={formData.password}
              onChange={handleChange}
              disabled={isSubmitting}
              error={errors.password}
              rightSlot={
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground/60 hover:text-foreground transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              }
            />
            <PasswordStrength password={formData.password} />

            <Field
              id="confirmPassword"
              name="confirmPassword"
              label="Confirm password"
              type={showConfirm ? 'text' : 'password'}
              autoComplete="new-password"
              placeholder="••••••••"
              icon={Lock}
              value={formData.confirmPassword}
              onChange={handleChange}
              disabled={isSubmitting}
              error={errors.confirmPassword}
              rightSlot={
                <button
                  type="button"
                  onClick={() => setShowConfirm((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground/60 hover:text-foreground transition-colors"
                >
                  {showConfirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              }
            />

            {errors.form && (
              <div className="p-3.5 rounded-xl bg-destructive/10 border border-destructive/25 text-destructive text-sm flex items-start gap-2.5">
                <span className="w-1.5 h-1.5 rounded-full bg-destructive mt-1.5 shrink-0" />
                <span>{errors.form}</span>
              </div>
            )}

            <Button
              type="submit"
              disabled={isSubmitting}
              className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl transition-all shadow-md shadow-primary/20"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Creating account & sending code...
                </>
              ) : (
                <>
                  Create Account
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          </form>
        )}

        {/* ─── STEP 2: EMAIL OTP VERIFICATION ─── */}
        {step === 'verify_email' && (
          <form onSubmit={handleVerifyOtpSubmit} className="space-y-5" noValidate>
            <div className="flex items-center justify-between p-3 bg-muted/20 border border-border rounded-xl">
              <div className="flex items-center gap-2.5 overflow-hidden">
                <Mail className="w-4 h-4 text-primary shrink-0" />
                <span className="text-sm font-medium text-foreground truncate">{pendingEmail}</span>
              </div>
              <button
                type="button"
                onClick={handleEditEmail}
                className="text-xs text-primary hover:underline font-medium shrink-0 ml-2"
              >
                Change
              </button>
            </div>

            <div className="space-y-2">
              <Label htmlFor="otpCode" className="text-sm font-medium text-muted-foreground">
                Enter 6-Digit Verification Code
              </Label>
              <div className="relative group">
                <KeyRound className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
                <Input
                  id="otpCode"
                  name="otpCode"
                  type="text"
                  autoFocus
                  autoComplete="one-time-code"
                  placeholder="000000"
                  value={otpCode}
                  maxLength={6}
                  onChange={(e) => {
                    const clean = e.target.value.replace(/\D/g, '')
                    setOtpCode(clean)
                    if (errors.otp) setErrors((prev) => ({ ...prev, otp: undefined }))
                  }}
                  className={`pl-10 h-12 text-center tracking-widest text-xl font-mono bg-muted/10 border-border/80 text-foreground
                    focus-visible:border-primary ${errors.otp ? 'border-destructive/65' : ''}`}
                  disabled={isSubmitting}
                />
              </div>
              <FieldError msg={errors.otp} />
              {resendFeedback && (
                <p className="text-xs text-emerald-400 mt-1 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                  {resendFeedback}
                </p>
              )}
            </div>

            <Button
              type="submit"
              disabled={isSubmitting || otpCode.trim().length < 6}
              className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl transition-all shadow-md shadow-primary/20"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Verifying code...
                </>
              ) : (
                <>
                  Verify Email
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>

            {/* Resend Cooldown Section */}
            <div className="pt-2 text-center">
              <p className="text-xs text-muted-foreground">
                Didn't receive the code?{' '}
                {resendCooldown > 0 ? (
                  <span className="text-muted-foreground/70 font-medium">
                    Resend in {formatCooldown(resendCooldown)}
                  </span>
                ) : (
                  <button
                    type="button"
                    onClick={handleResendOtp}
                    disabled={isResending}
                    className="text-primary hover:underline font-semibold inline-flex items-center gap-1"
                  >
                    {isResending ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <RotateCcw className="w-3 h-3" />
                    )}
                    Resend verification code
                  </button>
                )}
              </p>
            </div>
          </form>
        )}

        {/* ─── STEP 3: VERIFIED SUCCESS SCREEN ─── */}
        {step === 'verified' && (
          <div className="text-center py-4 space-y-6 animate-in fade-in zoom-in-95 duration-300">
            <div className="w-16 h-16 mx-auto rounded-full bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <CheckCircle2 className="w-8 h-8 text-emerald-400" />
            </div>

            <div className="space-y-2">
              <h3 className="text-xl font-bold text-foreground">Email Successfully Verified!</h3>
              <p className="text-sm text-muted-foreground max-w-sm mx-auto">
                Your email address has been verified. You can now sign in using your email and password.
              </p>
            </div>

            <Button
              type="button"
              onClick={() => navigate('/login?verified=true')}
              className="w-full h-11 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-xl transition-all shadow-md shadow-emerald-600/30"
            >
              Proceed to Sign In
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        )}
      </div>

      {step === 'register' && (
        <p className="mt-6 text-center text-sm text-muted-foreground">
          Already have an account?{' '}
          <Link
            to="/login"
            className="font-semibold text-primary hover:text-primary/80 transition-colors"
          >
            Sign in
          </Link>
        </p>
      )}
    </div>
  )
}
