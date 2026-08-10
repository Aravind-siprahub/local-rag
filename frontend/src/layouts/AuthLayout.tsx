import React from 'react'
import { Outlet } from 'react-router-dom'
import { FileText, ShieldCheck, Database, Cpu } from 'lucide-react'

export const AuthLayout: React.FC = () => {
  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-100 flex flex-col lg:flex-row relative overflow-hidden font-sans selection:bg-indigo-500 selection:text-white">
      {/* Background Animated Gradient Mesh */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-indigo-600/20 blur-[120px] animate-pulse" />
        <div className="absolute top-1/2 right-0 w-125 h-125 rounded-full bg-violet-600/15 blur-[150px]" />
        <div className="absolute -bottom-20 left-1/3 w-80 h-80 rounded-full bg-blue-600/20 blur-[100px]" />
      </div>

      {/* Main Content Area (Form Area) */}
      <div className="flex-1 flex flex-col justify-center items-center p-4 sm:p-8 z-10 min-h-screen lg:min-h-0">
        <div className="w-full max-w-md my-auto">
          <Outlet />
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-xs text-slate-400">
          <p>© {new Date().getFullYear()} Talk to My Data Inc. All rights reserved.</p>
        </footer>
      </div>

      {/* Feature Showcase Side Panel (Desktop View) */}
      <div className="hidden lg:flex flex-1 relative bg-linear-to-br from-slate-900/90 via-indigo-950/40 to-slate-950 p-12 border-l border-slate-800/80 backdrop-blur-2xl flex-col justify-between overflow-hidden">
        {/* Glow Overlay */}
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        {/* Top Header */}
        <div className="flex items-center gap-3 z-10">
          <div className="w-10 h-10 rounded-xl bg-linear-to-tr from-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
            <Database className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold tracking-tight text-white">Talk to My Data</h2>
            <p className="text-xs text-indigo-400 font-medium">Enterprise Local RAG & Vector Intelligence</p>
          </div>
        </div>

        {/* Middle Feature Showcase Card */}
        <div className="my-auto z-10 space-y-6 max-w-lg">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-medium">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>Next-Gen RAG Architecture</span>
          </div>

          <h3 className="text-3xl font-extrabold tracking-tight text-white leading-tight">
            Transform raw documents into actionable AI conversations.
          </h3>

          <p className="text-slate-400 text-sm leading-relaxed">
            Chat seamlessly with PDFs, Word documents, Excel spreadsheets, PowerPoint slides, and databases in seconds with complete privacy.
          </p>

          {/* Feature Highlights Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md">
              <FileText className="w-5 h-5 text-indigo-400 mb-2" />
              <h4 className="text-xs font-semibold text-white mb-1">Multi-Format Parsing</h4>
              <p className="text-[11px] text-slate-400">PDF, DOCX, XLSX, PPTX, CSV, and JSON parsing out of the box.</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md">
              <ShieldCheck className="w-5 h-5 text-emerald-400 mb-2" />
              <h4 className="text-xs font-semibold text-white mb-1">Zero Data Leakage</h4>
              <p className="text-[11px] text-slate-400">Local embeddings & private storage keep data completely secure.</p>
            </div>
          </div>
        </div>

        {/* Testimonial Quote */}
        <div className="pt-6 border-t border-slate-800/80 z-10">
          <blockquote className="text-xs text-slate-400 italic">
            "Talk to My Data cut our document research time by 80%. We can search across 5,000+ deployment guides instantly."
          </blockquote>
          <div className="flex items-center gap-2.5 mt-3">
            <div className="w-7 h-7 rounded-full bg-linear-to-r from-indigo-500 to-purple-500 flex items-center justify-center text-white text-xs font-bold">
              A
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-200">Aravind S.</p>
              <p className="text-[10px] text-slate-400">Lead AI Solutions Engineer</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
