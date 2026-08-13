import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Eye, EyeOff, Database, Mail, Lock, ArrowRight, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { AuthStore } from '@/features/auth/utils/authStore'
import { authApi } from '@/features/auth/api/authApi'

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

  const [formData, setFormData] = useState({ email: '', password: '', remember: false })
  const [errors, setErrors] = useState<{ email?: string; password?: string; form?: string }>({})
  const [showPassword, setShowPassword] = useState(false)
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
    if (!formData.email.trim()) errs.email = 'Email address is required'
    else if (!/\S+@\S+\.\S+/.test(formData.email)) errs.email = 'Enter a valid email address'
    if (!formData.password) errs.password = 'Password is required'
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
      const data = await authApi.login({
        email: formData.email.trim(),
        password: formData.password,
      })
      AuthStore.setSession(data.accessToken, data.user)
      AuthStore.setRememberedEmail(formData.email, formData.remember)
      window.dispatchEvent(new Event('auth:change'))
      void navigate('/')
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string }; status?: number } }
      const detail = e?.response?.data?.detail
      setErrors({
        form: typeof detail === 'string'
          ? detail.replace(/^Authentication failed:\s*/i, '')
          : 'Invalid email or password. Please try again.',
      })
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
          Welcome back
        </h1>
        <p className="mt-1.5 text-sm text-slate-400">
          Sign in to continue to your knowledge base
        </p>
      </div>

      {/* ── Glass Card ── */}
      <div className="relative rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl shadow-black/40 p-8 space-y-5">
        {/* subtle top highlight */}
        <div className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-indigo-500/50 to-transparent rounded-t-2xl" />

        <form onSubmit={handleSubmit} className="space-y-5" noValidate>
          {/* Email */}
          <div className="space-y-1.5">
            <Label htmlFor="email" className="text-sm font-medium text-slate-300">
              Email address
            </Label>
            <div className="relative group">
              <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-indigo-400 transition-colors" />
              <Input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                placeholder="name@example.com"
                value={formData.email}
                onChange={handleChange}
                className={`pl-10 bg-slate-900/60 border-slate-700/60 text-white placeholder:text-slate-500
                  focus-visible:border-indigo-500 focus-visible:ring-1 focus-visible:ring-indigo-500/50
                  hover:border-slate-600 transition-colors
                  ${errors.email ? 'border-red-500/70 focus-visible:border-red-500 focus-visible:ring-red-500/30' : ''}`}
                disabled={isSubmitting}
              />
            </div>
            <FieldError msg={errors.email} />
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label htmlFor="password" className="text-sm font-medium text-slate-300">
                Password
              </Label>
              <Link
                to="/forgot-password"
                tabIndex={-1}
                className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                Forgot password?
              </Link>
            </div>
            <div className="relative group">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-indigo-400 transition-colors" />
              <Input
                id="password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete="current-password"
                placeholder="••••••••"
                value={formData.password}
                onChange={handleChange}
                className={`pl-10 pr-10 bg-slate-900/60 border-slate-700/60 text-white placeholder:text-slate-500
                  focus-visible:border-indigo-500 focus-visible:ring-1 focus-visible:ring-indigo-500/50
                  hover:border-slate-600 transition-colors
                  ${errors.password ? 'border-red-500/70 focus-visible:border-red-500 focus-visible:ring-red-500/30' : ''}`}
                disabled={isSubmitting}
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                tabIndex={-1}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors focus:outline-none"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <FieldError msg={errors.password} />
          </div>

          {/* Remember me */}
          <div className="flex items-center gap-2.5">
            <input
              id="remember"
              name="remember"
              type="checkbox"
              checked={formData.remember}
              onChange={handleChange}
              className="w-4 h-4 rounded border-slate-600 bg-slate-800 accent-indigo-500 cursor-pointer"
            />
            <Label htmlFor="remember" className="text-sm text-slate-400 cursor-pointer select-none">
              Remember me for 30 days
            </Label>
          </div>

          {/* Form-level error (e.g. bad credentials from backend) */}
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
                Signing in…
              </>
            ) : (
              <>
                Sign in
                <ArrowRight className="w-4 h-4 ml-2" />
              </>
            )}
          </Button>
        </form>

        {/* ── Divider ── */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-px bg-slate-700/60" />
          <span className="text-xs text-slate-500 shrink-0">or</span>
          <div className="flex-1 h-px bg-slate-700/60" />
        </div>

        {/* ── Sign-up link ── */}
        <p className="text-center text-sm text-slate-400">
          Don't have an account?{' '}
          <Link
            to="/signup"
            className="font-semibold text-indigo-400 hover:text-indigo-300 transition-colors underline-offset-4 hover:underline"
          >
            Create one free
          </Link>
        </p>
      </div>
    </div>
  )
}
