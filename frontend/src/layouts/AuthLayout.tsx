import React from 'react'
import { Outlet } from 'react-router-dom'
import { FileText, ShieldCheck, Database, Cpu } from 'lucide-react'

export const AuthLayout: React.FC = () => {
  return (
    <div className="min-h-screen w-full bg-background text-foreground flex flex-col lg:flex-row relative overflow-hidden font-sans selection:bg-primary/20 selection:text-primary">
      {/* Background Animated Gradient Mesh */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-primary/10 blur-[120px] animate-pulse" />
        <div className="absolute top-1/2 right-0 w-125 h-125 rounded-full bg-primary/5 blur-[150px]" />
        <div className="absolute -bottom-20 left-1/3 w-80 h-80 rounded-full bg-primary/10 blur-[100px]" />
      </div>

      {/* Main Content Area (Form Area) */}
      <div className="flex-1 flex flex-col justify-center items-center p-4 sm:p-8 z-10 min-h-screen lg:min-h-0">
        <div className="w-full max-w-md my-auto">
          <Outlet />
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-xs text-muted-foreground">
          <p>© {new Date().getFullYear()} Talk to My Data Inc. All rights reserved.</p>
        </footer>
      </div>

      {/* Feature Showcase Side Panel (Desktop View) */}
      <div className="hidden lg:flex flex-1 relative bg-card p-12 border-l border-border backdrop-blur-2xl flex-col justify-between overflow-hidden">
        {/* Glow Overlay */}
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-primary/5 rounded-full blur-3xl pointer-events-none" />

        {/* Top Header */}
        <div className="flex items-center gap-3 z-10">
          <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-lg shadow-primary/25">
            <Database className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <h2 className="text-lg font-bold tracking-tight text-foreground font-display">Talk to My Data</h2>
            <p className="text-xs text-primary font-medium">Enterprise Local RAG & Vector Intelligence</p>
          </div>
        </div>

        {/* Middle Feature Showcase Card */}
        <div className="my-auto z-10 space-y-6 max-w-lg">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-medium">
            <Cpu className="w-3.5 h-3.5 text-primary" />
            <span>Next-Gen RAG Architecture</span>
          </div>

          <h3 className="text-3xl font-extrabold tracking-tight text-foreground leading-tight font-display">
            Transform raw documents into actionable AI conversations.
          </h3>

          <p className="text-muted-foreground text-sm leading-relaxed">
            Chat seamlessly with PDFs, Word documents, Excel spreadsheets, PowerPoint slides, and databases in seconds with complete privacy.
          </p>

          {/* Feature Highlights Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4">
            <div className="p-4 rounded-xl bg-muted/40 border border-border/80 backdrop-blur-md">
              <FileText className="w-5 h-5 text-primary mb-2" />
              <h4 className="text-xs font-semibold text-foreground mb-1">Multi-Format Parsing</h4>
              <p className="text-[11px] text-muted-foreground">PDF, DOCX, XLSX, PPTX, CSV, and JSON parsing out of the box.</p>
            </div>

            <div className="p-4 rounded-xl bg-muted/40 border border-border/80 backdrop-blur-md">
              <ShieldCheck className="w-5 h-5 text-success mb-2" />
              <h4 className="text-xs font-semibold text-foreground mb-1">Zero Data Leakage</h4>
              <p className="text-[11px] text-muted-foreground">Local embeddings & private storage keep data completely secure.</p>
            </div>
          </div>
        </div>

        {/* Testimonial Quote */}
        <div className="pt-6 border-t border-border z-10">
          <blockquote className="text-xs text-muted-foreground italic">
            "Talk to My Data cut our document research time by 80%. We can search across 5,000+ deployment guides instantly."
          </blockquote>
          <div className="flex items-center gap-2.5 mt-3">
            <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs font-bold">
              A
            </div>
            <div>
              <p className="text-xs font-semibold text-foreground">Aravind S.</p>
              <p className="text-[10px] text-muted-foreground">Lead AI Solutions Engineer</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
