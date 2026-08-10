import React from 'react'

interface SocialLoginButtonsProps {
  onGoogleLogin?: () => void
  onMicrosoftLogin?: () => void
  isLoading?: boolean
}

export const SocialLoginButtons: React.FC<SocialLoginButtonsProps> = ({
  onGoogleLogin,
  onMicrosoftLogin,
  isLoading = false,
}) => {
  return (
    <div className="space-y-3">
      {/* Divider */}
      <div className="relative flex items-center justify-center my-5">
        <div className="w-full border-t border-slate-800" />
        <span className="absolute px-3 bg-slate-900 text-[11px] font-medium text-slate-400 uppercase tracking-wider">
          Or continue with
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {/* Google Login Button */}
        <button
          type="button"
          disabled={isLoading}
          onClick={onGoogleLogin}
          className="flex items-center justify-center gap-2 py-2.5 px-4 bg-slate-950/70 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 rounded-xl text-xs font-medium text-slate-200 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-slate-700 disabled:opacity-50"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24">
            <path
              fill="#EA4335"
              d="M12 5c1.6 0 3 .6 4.1 1.6l3.1-3.1C17.3 1.7 14.8 1 12 1 7.5 1 3.7 3.6 1.9 7.3l3.7 2.9C6.5 7.3 9 5 12 5z"
            />
            <path
              fill="#4285F4"
              d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.5c-.3 1.5-1.1 2.8-2.4 3.7l3.7 2.9c2.2-2 3.7-5 3.7-8.8z"
            />
            <path
              fill="#FBBC05"
              d="M5.6 14.8c-.3-.8-.4-1.8-.4-2.8s.1-2 .4-2.8L1.9 6.3C.7 8.7 0 10.3 0 12s.7 3.3 1.9 5.7l3.7-2.9z"
            />
            <path
              fill="#34A853"
              d="M12 23c3.2 0 6-1.1 8-3l-3.7-2.9c-1.1.7-2.5 1.2-4.3 1.2-3 0-5.5-2.3-6.4-5.2L1.9 16C3.7 19.7 7.5 23 12 23z"
            />
          </svg>
          <span>Google</span>
        </button>

        {/* Microsoft Login Button */}
        <button
          type="button"
          disabled={isLoading}
          onClick={onMicrosoftLogin}
          className="flex items-center justify-center gap-2 py-2.5 px-4 bg-slate-950/70 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 rounded-xl text-xs font-medium text-slate-200 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-slate-700 disabled:opacity-50"
        >
          <svg className="w-4 h-4" viewBox="0 0 23 23">
            <path fill="#f35325" d="M1 1h10v10H1z" />
            <path fill="#81bc06" d="M12 1h10v10H12z" />
            <path fill="#05a6f0" d="M1 12h10v10H1z" />
            <path fill="#ffba08" d="M12 12h10v10H12z" />
          </svg>
          <span>Microsoft</span>
        </button>
      </div>
    </div>
  )
}
