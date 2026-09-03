import React, { useState, useMemo, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Database,
  Mail,
  Lock,
  ArrowRight,
  ArrowLeft,
  Loader2,
  CheckCircle2,
  XCircle,
  KeyRound,
  RotateCcw,
  Eye,
  EyeOff,
  ShieldCheck,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { authApi } from '@/features/auth/api/authApi'

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

export function ForgotPassword() {
  const navigate = useNavigate()

  const [step, setStep] = useState<'enter_email' | 'enter_otp' | 'success'>('enter_email')
  const [email, setEmail] = useState('')
  const [otpCode, setOtpCode] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [resendCooldown, setResendCooldown] = useState(0)
  const [feedbackNotice, setFeedbackNotice] = useState<string | null>(null)
  const [errors, setErrors] = useState<{
    email?: string
    otp?: string
    password?: string
    confirmPassword?: string
    form?: string
  }>({})

  /* ── 60-second cooldown timer for resending OTP ── */
  useEffect(() => {
    if (resendCooldown <= 0) return
    const timer = setInterval(() => {
      setResendCooldown((prev) => (prev <= 1 ? 0 : prev - 1))
    }, 1000)
    return () => clearInterval(timer)
  }, [resendCooldown])

  /* ── password strength score ── */
  const passedCount = useMemo(() => {
    if (!newPassword) return 0
    return RULES.filter((r) => r.test(newPassword)).length
  }, [newPassword])

  /* ── submit step 1: request reset code ── */
  const handleRequestCode = async (e: React.FormEvent) => {
    e.preventDefault()
    const cleanEmail = email.trim().toLowerCase()
    if (!cleanEmail) {
      setErrors({ email: 'Email address is required' })
      return
    }
    if (!/\S+@\S+\.\S+/.test(cleanEmail)) {
      setErrors({ email: 'Enter a valid email address' })
      return
    }

    setIsSubmitting(true)
    setErrors({})
    setFeedbackNotice(null)

    try {
      const res = await authApi.forgotPassword({ email: cleanEmail })
      setFeedbackNotice(res.message)
      setResendCooldown(300)
      setStep('enter_otp')
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string }; status?: number } }
      const detail = e?.response?.data?.detail
      setErrors({
        form: typeof detail === 'string' ? detail : 'Failed to send reset code. Please try again.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  /* ── resend reset code ── */
  const handleResend = async () => {
    if (resendCooldown > 0 || isSubmitting) return
    setIsSubmitting(true)
    setFeedbackNotice(null)
    setErrors({})

    try {
      const res = await authApi.forgotPassword({ email: email.trim().toLowerCase() })
      setFeedbackNotice(res.message || 'A fresh reset code has been sent to your email.')
      setResendCooldown(300)
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string }; status?: number } }
      const detail = e?.response?.data?.detail
      setErrors({
        form: typeof detail === 'string' ? detail : 'Unable to resend reset code. Please try again later.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  /* ── submit step 2: verify OTP and set new password ── */
  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault()
    const errs: typeof errors = {}

    const cleanCode = otpCode.trim()
    if (!cleanCode || cleanCode.length < 4) {
      errs.otp = 'Please enter the 6-digit reset code from your email'
    }

    if (!newPassword) {
      errs.password = 'New password is required'
    } else if (newPassword.length < 8) {
      errs.password = 'Password must be at least 8 characters'
    } else if (passedCount < RULES.length) {
      errs.password = 'Password must meet all 5 criteria below (including uppercase letter)'
    }

    if (!confirmPassword) {
      errs.confirmPassword = 'Confirm your new password'
    } else if (newPassword !== confirmPassword) {
      errs.confirmPassword = 'Passwords do not match'
    }

    if (Object.keys(errs).length > 0) {
      setErrors(errs)
      return
    }

    setIsSubmitting(true)
    setErrors({})
    setFeedbackNotice(null)

    try {
      await authApi.resetPassword({
        email: email.trim().toLowerCase(),
        code: cleanCode,
        new_password: newPassword,
      })
      setStep('success')
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: unknown }; status?: number } }
      const rawDetail = e?.response?.data?.detail
      let errorMsg = 'Failed to reset password. Please verify the code and password requirements.'

      if (typeof rawDetail === 'string') {
        errorMsg = rawDetail
      } else if (Array.isArray(rawDetail) && rawDetail.length > 0) {
        errorMsg = rawDetail
          .map((d: { msg?: string }) => (d.msg || '').replace(/^Value error,\s*/i, ''))
          .filter(Boolean)
          .join('; ')
      }

      setErrors({ form: errorMsg })
    } finally {
      setIsSubmitting(false)
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
          {step === 'enter_email' && 'Reset your password'}
          {step === 'enter_otp' && 'Set new password'}
          {step === 'success' && 'Password updated!'}
        </h1>
        <p className="mt-1.5 text-sm text-muted-foreground">
          {step === 'enter_email' && "Enter your email address and we'll send you a 6-digit code to reset your password."}
          {step === 'enter_otp' && `We sent a 6-digit code to ${email}. Enter the code and choose a new password.`}
          {step === 'success' && 'Your password has been changed successfully.'}
        </p>
      </div>

      {/* ── Card ── */}
      <div className="relative rounded-2xl border border-border bg-card shadow-lg p-8 space-y-5">
        <div className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-primary/50 to-transparent rounded-t-2xl" />

        {/* Global form error */}
        {errors.form && (
          <div className="p-3.5 rounded-xl bg-destructive/10 border border-destructive/25 text-destructive text-sm flex items-start gap-2.5">
            <span className="w-1.5 h-1.5 rounded-full bg-destructive mt-1.5 shrink-0" />
            <span>{errors.form}</span>
          </div>
        )}

        {/* Info feedback */}
        {feedbackNotice && step === 'enter_otp' && (
          <div className="p-3.5 rounded-xl bg-sky-500/10 border border-sky-500/30 text-sky-300 text-sm flex items-center gap-2.5">
            <ShieldCheck className="w-4 h-4 shrink-0 text-sky-400" />
            <span>{feedbackNotice}</span>
          </div>
        )}

        {/* ══════════ STEP 1: Enter Email ══════════ */}
        {step === 'enter_email' && (
          <form onSubmit={handleRequestCode} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="email" className="text-sm font-medium text-muted-foreground">
                Email address
              </Label>
              <div className="relative group">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value)
                    if (errors.email) setErrors((prev) => ({ ...prev, email: undefined, form: undefined }))
                  }}
                  className={`pl-10 h-10 bg-muted/10 border-border/80 text-foreground placeholder:text-muted-foreground/45
                    focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary/30
                    hover:border-border transition-colors
                    ${errors.email ? 'border-destructive/65 focus-visible:border-destructive' : ''}`}
                  disabled={isSubmitting}
                  autoFocus
                />
              </div>
              <FieldError msg={errors.email} />
            </div>

            <Button
              type="submit"
              disabled={isSubmitting}
              className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl transition-all shadow-md shadow-primary/20"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Sending code...
                </>
              ) : (
                <>
                  Send Reset Code
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>

            <div className="pt-2 text-center">
              <Link
                to="/login"
                className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Back to Sign in
              </Link>
            </div>
          </form>
        )}

        {/* ══════════ STEP 2: Enter OTP & New Password ══════════ */}
        {step === 'enter_otp' && (
          <form onSubmit={handleResetPassword} className="space-y-4">
            {/* Display email + Change button */}
            <div className="flex items-center justify-between p-3 rounded-xl bg-muted/20 border border-border/60">
              <div className="flex items-center gap-2 overflow-hidden">
                <Mail className="w-4 h-4 text-primary shrink-0" />
                <span className="text-sm font-medium text-foreground truncate">{email}</span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setStep('enter_email')
                  setErrors({})
                }}
                className="text-xs text-primary hover:underline font-medium shrink-0 ml-2"
              >
                Change
              </button>
            </div>

            {/* 6-Digit OTP code */}
            <div className="space-y-1.5">
              <Label htmlFor="otpCode" className="text-sm font-medium text-muted-foreground">
                Enter 6-Digit Reset Code
              </Label>
              <div className="relative group">
                <KeyRound className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
                <Input
                  id="otpCode"
                  type="text"
                  maxLength={6}
                  placeholder="000000"
                  value={otpCode}
                  onChange={(e) => {
                    const val = e.target.value.replace(/\D/g, '')
                    setOtpCode(val)
                    if (errors.otp) setErrors((prev) => ({ ...prev, otp: undefined, form: undefined }))
                  }}
                  className={`pl-10 tracking-widest font-mono text-center text-lg font-bold h-11 bg-muted/10 border-border/80 text-foreground
                    focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary/30
                    ${errors.otp ? 'border-destructive/65 focus-visible:border-destructive' : ''}`}
                  disabled={isSubmitting}
                  autoFocus
                />
              </div>
              <FieldError msg={errors.otp} />
            </div>

            {/* New Password */}
            <div className="space-y-1.5">
              <Label htmlFor="newPassword" className="text-sm font-medium text-muted-foreground">
                New Password
              </Label>
              <div className="relative group">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
                <Input
                  id="newPassword"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="••••••••"
                  value={newPassword}
                  onChange={(e) => {
                    setNewPassword(e.target.value)
                    if (errors.password) setErrors((prev) => ({ ...prev, password: undefined, form: undefined }))
                  }}
                  className={`pl-10 pr-10 h-10 bg-muted/10 border-border/80 text-foreground placeholder:text-muted-foreground/45
                    focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary/30
                    ${errors.password ? 'border-destructive/65 focus-visible:border-destructive' : ''}`}
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground/60 hover:text-foreground transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <FieldError msg={errors.password} />

              {/* Password strength bar */}
              {newPassword && (
                <div className="space-y-2 pt-1">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-muted-foreground">Strength:</span>
                    <span className={`font-semibold ${STRENGTH_TEXT[passedCount]}`}>
                      {STRENGTH_LABELS[passedCount]}
                    </span>
                  </div>
                  <div className="flex gap-1 h-1.5">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <div
                        key={i}
                        className={`flex-1 rounded-full transition-all duration-300 ${
                          i <= passedCount ? STRENGTH_COLORS[passedCount] : 'bg-muted/40'
                        }`}
                      />
                    ))}
                  </div>
                  {/* Criteria list */}
                  <div className="grid grid-cols-1 gap-1 pt-1">
                    {RULES.map((rule) => {
                      const ok = rule.test(newPassword)
                      return (
                        <div key={rule.label} className="flex items-center gap-1.5 text-xs">
                          {ok ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                          ) : (
                            <XCircle className="w-3.5 h-3.5 text-muted-foreground/40 shrink-0" />
                          )}
                          <span className={ok ? 'text-foreground/80' : 'text-muted-foreground/60'}>
                            {rule.label}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div className="space-y-1.5">
              <Label htmlFor="confirmPassword" className="text-sm font-medium text-muted-foreground">
                Confirm New Password
              </Label>
              <div className="relative group">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground/60 group-focus-within:text-primary transition-colors" />
                <Input
                  id="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  placeholder="••••••••"
                  value={confirmPassword}
                  onChange={(e) => {
                    setConfirmPassword(e.target.value)
                    if (errors.confirmPassword) setErrors((prev) => ({ ...prev, confirmPassword: undefined, form: undefined }))
                  }}
                  className={`pl-10 pr-10 h-10 bg-muted/10 border-border/80 text-foreground placeholder:text-muted-foreground/45
                    focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary/30
                    ${errors.confirmPassword ? 'border-destructive/65 focus-visible:border-destructive' : ''}`}
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground/60 hover:text-foreground transition-colors"
                >
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <FieldError msg={errors.confirmPassword} />
            </div>

            {/* Submit */}
            <Button
              type="submit"
              disabled={isSubmitting}
              className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl transition-all shadow-md shadow-primary/20"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Updating password...
                </>
              ) : (
                <>
                  Reset Password
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>

            {/* Resend code */}
            <div className="flex items-center justify-between text-xs text-muted-foreground pt-2">
              <span>Didn't get the code?</span>
              <button
                type="button"
                onClick={handleResend}
                disabled={resendCooldown > 0 || isSubmitting}
                className="text-primary hover:underline font-medium inline-flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RotateCcw className={`w-3.5 h-3.5 ${isSubmitting ? 'animate-spin' : ''}`} />
                {resendCooldown > 0 ? `Resend code (${formatCooldown(resendCooldown)})` : 'Resend code'}
              </button>
            </div>
          </form>
        )}

        {/* ══════════ STEP 3: Success Screen ══════════ */}
        {step === 'success' && (
          <div className="text-center py-4 space-y-5">
            <div className="w-14 h-14 rounded-full bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center mx-auto text-emerald-400 shadow-lg shadow-emerald-500/20">
              <CheckCircle2 className="w-8 h-8" />
            </div>

            <div className="space-y-2">
              <h2 className="text-xl font-bold text-foreground">Password Reset Successfully!</h2>
              <p className="text-sm text-muted-foreground max-w-sm mx-auto">
                Your account password has been updated. You can now log in with your new credentials.
              </p>
            </div>

            <Button
              type="button"
              onClick={() => navigate('/login?reset=true')}
              className="w-full h-10 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl transition-all shadow-md shadow-primary/20"
            >
              Proceed to Sign In
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <div className="mt-8 text-center text-xs text-muted-foreground">
        Remember your password?{' '}
        <Link to="/login" className="text-primary hover:underline font-medium">
          Sign in
        </Link>
      </div>
    </div>
  )
}
