import React, { useState, useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Eye, EyeOff, Database, Mail, Lock, User, ArrowRight, Loader2,
  CheckCircle2, XCircle,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { AuthStore } from '@/features/auth/utils/authStore'
import { authApi } from '@/features/auth/api/authApi'
import { createUser } from '@/services/users.service'

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

/* ── password strength ── */
interface StrengthRule { label: string; test: (pw: string) => boolean }
const RULES: StrengthRule[] = [
  { label: 'At least 8 characters', test: (pw) => pw.length >= 8 },
  { label: 'Uppercase letter', test: (pw) => /[A-Z]/.test(pw) },
  { label: 'Lowercase letter', test: (pw) => /[a-z]/.test(pw) },
  { label: 'Number', test: (pw) => /[0-9]/.test(pw) },
  { label: 'Special character (!@#$%^&* etc.)', test: (pw) => /[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;/`~]/.test(pw) },
]

const STRENGTH_LABELS = ['', 'Very Weak', 'Weak', 'Fair', 'Good', 'Strong'] as const
const STRENGTH_COLORS = ['', 'bg-red-600', 'bg-red-500', 'bg-amber-500', 'bg-yellow-400', 'bg-emerald-500'] as const
const STRENGTH_TEXT   = ['', 'text-red-500', 'text-red-400', 'text-amber-400', 'text-yellow-400', 'text-emerald-400'] as const

function PasswordStrength({ password }: { password: string }) {
  const strength = useMemo(() => RULES.filter((r) => r.test(password)).length, [password])
  if (!password) return null
  return (
    <div className="mt-2.5 space-y-2 animate-in fade-in slide-in-from-top-1 duration-200">
      {/* bar */}
      <div className="flex gap-1">
        {RULES.map((_, i) => (
          <div
            key={i}
            className={`h-1 flex-1 rounded-full transition-all duration-300 ${
              i < strength ? STRENGTH_COLORS[strength] : 'bg-slate-700'
            }`}
          />
        ))}
      </div>
      <p className={`text-xs font-medium ${STRENGTH_TEXT[strength]}`}>
        {STRENGTH_LABELS[strength]}
      </p>
      {/* rule checklist */}
      <ul className="space-y-1">
        {RULES.map((rule) => {
          const ok = rule.test(password)
          return (
            <li key={rule.label} className="flex items-center gap-1.5 text-[11px]">
              {ok
                ? <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                : <XCircle className="w-3 h-3 text-slate-600 shrink-0" />}
              <span className={ok ? 'text-slate-300' : 'text-slate-500'}>{rule.label}</span>
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
}

function Field({
  id, name, label, type = 'text', placeholder, icon: Icon, value, onChange, disabled, rightSlot, error,
}: FieldProps) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="text-sm font-medium text-slate-300">{label}</Label>
      <div className="relative group">
        <Icon className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-indigo-400 transition-colors" />
        <Input
          id={id}
          name={name}
          type={type}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          disabled={disabled}
          className={`pl-10 ${rightSlot ? 'pr-10' : ''} bg-slate-900/60 border-slate-700/60 text-white placeholder:text-slate-500
            focus-visible:border-indigo-500 focus-visible:ring-1 focus-visible:ring-indigo-500/50
            hover:border-slate-600 transition-colors
            ${error ? 'border-red-500/70 focus-visible:border-red-500 focus-visible:ring-red-500/30' : ''}`}
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
    acceptTerms: false,
  })

  const [errors, setErrors] = useState<{
    fullName?: string
    email?: string
    password?: string
    confirmPassword?: string
    acceptTerms?: string
    form?: string
  }>({})

  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

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
    else if (!RULES.every((r) => r.test(formData.password)))
      errs.password = 'Password does not meet all requirements'
    if (!formData.confirmPassword) errs.confirmPassword = 'Please confirm your password'
    else if (formData.password !== formData.confirmPassword)
      errs.confirmPassword = 'Passwords do not match'
    if (!formData.acceptTerms) errs.acceptTerms = 'You must accept the terms to continue'
    setErrors(errs)
    return Object.keys(errs).length === 0
  }

  /* ── submit ── */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    setIsSubmitting(true)
    setErrors({})

    try {
      // 1. Create user account via backend
      await createUser({
        email: formData.email.trim(),
        password: formData.password,
        full_name: formData.fullName.trim(),
        role: 'member',
      })

      // 2. Immediately login to get a real signed JWT
      const data = await authApi.login({
        email: formData.email.trim(),
        password: formData.password,
      })
      AuthStore.setSession(data.accessToken, data.user)
      window.dispatchEvent(new Event('auth:change'))
      void navigate('/')
    } catch (err: unknown) {
      const e = err as { response?: { status?: number; data?: { detail?: string } } }
      const status = e?.response?.status
      const detail = e?.response?.data?.detail

      if (status === 409) {
        // Email already registered — redirect to login
        setErrors({ form: 'An account with this email already exists. Please sign in.' })
      } else if (status === 422) {
        // Backend validation failed (e.g. password too weak)
        const msg = typeof detail === 'string'
          ? detail.replace(/^Value error,\s*/i, '')
          : 'Invalid signup data. Please check your inputs.'
        setErrors({ form: msg })
      } else {
        setErrors({
          form: typeof detail === 'string'
            ? detail
            : 'Account creation failed. Please try again.',
        })
      }
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
        <h1 className="text-3xl font-extrabold text-white tracking-tight font-display leading-tight">
          Create your account
        </h1>
        <p className="mt-1.5 text-sm text-slate-400">
          Start querying your documents with local AI in minutes
        </p>
      </div>

      {/* ── Glass Card ── */}
      <div className="relative rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl shadow-black/40 p-8 space-y-5">
        {/* top highlight */}
        <div className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-violet-500/50 to-transparent rounded-t-2xl" />

        <form onSubmit={handleSubmit} className="space-y-4" noValidate>
          {/* Full Name */}
          <Field
            id="fullName" name="fullName"
            label="Full name" placeholder="Aravind S."
            icon={User} value={formData.fullName} onChange={handleChange} disabled={isSubmitting} error={errors.fullName}
          />

          {/* Email */}
          <Field
            id="email" name="email"
            label="Email address" type="email" placeholder="name@example.com"
            icon={Mail} value={formData.email} onChange={handleChange} disabled={isSubmitting} error={errors.email}
          />

          {/* Password */}
          <div className="space-y-1.5">
            <Label htmlFor="password" className="text-sm font-medium text-slate-300">Password</Label>
            <div className="relative group">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-indigo-400 transition-colors" />
              <Input
                id="password" name="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete="new-password"
                placeholder="••••••••"
                value={formData.password}
                onChange={handleChange}
                disabled={isSubmitting}
                className={`pl-10 pr-10 bg-slate-900/60 border-slate-700/60 text-white placeholder:text-slate-500
                  focus-visible:border-indigo-500 focus-visible:ring-1 focus-visible:ring-indigo-500/50
                  hover:border-slate-600 transition-colors
                  ${errors.password ? 'border-red-500/70 focus-visible:border-red-500 focus-visible:ring-red-500/30' : ''}`}
              />
              <button
                type="button" tabIndex={-1}
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors focus:outline-none"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <FieldError msg={errors.password} />
            <PasswordStrength password={formData.password} />
          </div>

          {/* Confirm Password */}
          <div className="space-y-1.5">
            <Label htmlFor="confirmPassword" className="text-sm font-medium text-slate-300">
              Confirm password
            </Label>
            <div className="relative group">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-indigo-400 transition-colors" />
              <Input
                id="confirmPassword" name="confirmPassword"
                type={showConfirm ? 'text' : 'password'}
                autoComplete="new-password"
                placeholder="••••••••"
                value={formData.confirmPassword}
                onChange={handleChange}
                disabled={isSubmitting}
                className={`pl-10 pr-10 bg-slate-900/60 border-slate-700/60 text-white placeholder:text-slate-500
                  focus-visible:border-indigo-500 focus-visible:ring-1 focus-visible:ring-indigo-500/50
                  hover:border-slate-600 transition-colors
                  ${errors.confirmPassword ? 'border-red-500/70 focus-visible:border-red-500 focus-visible:ring-red-500/30' : ''}`}
              />
              <button
                type="button" tabIndex={-1}
                onClick={() => setShowConfirm((v) => !v)}
                aria-label={showConfirm ? 'Hide password' : 'Show password'}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors focus:outline-none"
              >
                {showConfirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {/* Passwords match indicator */}
            {formData.confirmPassword && (
              <p className={`flex items-center gap-1 text-xs mt-1 ${
                formData.password === formData.confirmPassword
                  ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {formData.password === formData.confirmPassword
                  ? <><CheckCircle2 className="w-3 h-3" /> Passwords match</>
                  : <><XCircle className="w-3 h-3" /> Passwords do not match</>}
              </p>
            )}
            <FieldError msg={errors.confirmPassword} />
          </div>

          {/* Terms */}
          <div>
            <div className="flex items-start gap-2.5">
              <input
                id="acceptTerms"
                name="acceptTerms"
                type="checkbox"
                checked={formData.acceptTerms}
                onChange={handleChange}
                disabled={isSubmitting}
                className="mt-0.5 w-4 h-4 rounded border-slate-600 bg-slate-800 accent-indigo-500 cursor-pointer shrink-0"
              />
              <Label htmlFor="acceptTerms" className="text-sm text-slate-400 cursor-pointer select-none leading-relaxed">
                I agree to the{' '}
                <a href="#" className="text-indigo-400 hover:text-indigo-300 underline-offset-2 hover:underline">
                  Terms of Service
                </a>{' '}
                and{' '}
                <a href="#" className="text-indigo-400 hover:text-indigo-300 underline-offset-2 hover:underline">
                  Privacy Policy
                </a>
              </Label>
            </div>
            <FieldError msg={errors.acceptTerms} />
          </div>

          {/* Form-level error (backend 422 / 409 / auth failure) */}
          {errors.form && (
            <p className="flex items-center gap-1.5 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 animate-in fade-in slide-in-from-top-1 duration-200">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-red-400 shrink-0" />
              {errors.form}
            </p>
          )}

          {/* Submit */}
          <Button
            type="submit"
            disabled={isSubmitting}
            className="w-full h-11 bg-linear-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500
              text-white font-semibold rounded-xl shadow-lg shadow-indigo-500/25
              transition-all duration-200 hover:shadow-indigo-500/40 hover:scale-[1.01]
              disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:scale-100 mt-1"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Creating account…
              </>
            ) : (
              <>
                Create account
                <ArrowRight className="w-4 h-4 ml-2" />
              </>
            )}
          </Button>
        </form>

        {/* ── Sign-in link ── */}
        <p className="text-center text-sm text-slate-400 pt-1">
          Already have an account?{' '}
          <Link
            to="/login"
            className="font-semibold text-indigo-400 hover:text-indigo-300 transition-colors underline-offset-4 hover:underline"
          >
            Sign in
          </Link>
        </p>
      </div>
    </div>
  )
}
