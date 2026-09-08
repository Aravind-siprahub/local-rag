import { useState, useRef, useEffect, useCallback } from 'react'

export type SpeechRecognitionStatus = 'idle' | 'recording' | 'processing' | 'error'

export interface LanguageOption {
  code: string
  label: string
  shortLabel: string
}

export const SUPPORTED_LANGUAGES: LanguageOption[] = [
  { code: 'en-US', label: 'English', shortLabel: 'EN' },
]

export interface UseSpeechRecognitionOptions {
  onTranscript?: (transcript: string, isFinal: boolean) => void
  onError?: (error: string) => void
  onCancel?: () => void
  defaultLanguage?: string
}

export interface UseSpeechRecognitionReturn {
  isSupported: boolean
  status: SpeechRecognitionStatus
  transcript: string
  interimTranscript: string
  errorMessage: string | null
  language: string
  supportedLanguages: LanguageOption[]
  startListening: () => void
  stopListening: () => void
  cancelListening: () => void
  setLanguage: (lang: string) => void
  clearError: () => void
}

// Helper to get SpeechRecognition constructor across browsers
function getSpeechRecognitionClass(): (new () => any) | null {
  const globalScope = typeof globalThis !== 'undefined' ? (globalThis as any) : null
  if (!globalScope) return null
  return (
    globalScope.SpeechRecognition ||
    globalScope.webkitSpeechRecognition ||
    (globalScope.window && (globalScope.window.SpeechRecognition || globalScope.window.webkitSpeechRecognition)) ||
    null
  )
}

function cleanupRecognitionInstance(inst: any) {
  if (!inst) return
  try {
    inst.onresult = null
    inst.onerror = null
    inst.onend = null
    inst.onstart = null
    inst.abort()
  } catch (e) {
    // Ignore abort errors
  }
}

export function useSpeechRecognition({
  onTranscript,
  onError,
  onCancel,
  defaultLanguage = 'en-US',
}: UseSpeechRecognitionOptions = {}): UseSpeechRecognitionReturn {
  const [isSupported, setIsSupported] = useState<boolean>(false)
  const [status, setStatus] = useState<SpeechRecognitionStatus>('idle')
  const [transcript, setTranscript] = useState<string>('')
  const [interimTranscript, setInterimTranscript] = useState<string>('')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [language, setLanguageState] = useState<string>(defaultLanguage)

  const recognitionRef = useRef<any>(null)
  const isCancelledRef = useRef<boolean>(false)
  const isRecordingRef = useRef<boolean>(false)
  const restartTimerRef = useRef<any>(null)
  const accumulatedTranscriptRef = useRef<string>('')

  const clearRestartTimer = useCallback(() => {
    if (restartTimerRef.current) {
      clearTimeout(restartTimerRef.current)
      restartTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    const SpeechRecognitionClass = getSpeechRecognitionClass()
    setIsSupported(Boolean(SpeechRecognitionClass))

    return () => {
      isCancelledRef.current = true
      isRecordingRef.current = false
      clearRestartTimer()
      const inst = recognitionRef.current
      recognitionRef.current = null
      cleanupRecognitionInstance(inst)
    }
  }, [clearRestartTimer])

  const clearError = useCallback(() => {
    setErrorMessage(null)
    if (status === 'error') {
      setStatus('idle')
    }
  }, [status])

  const setLanguage = useCallback((lang: string) => {
    setLanguageState(lang)
    if (recognitionRef.current) {
      recognitionRef.current.lang = lang
    }
  }, [])

  const cancelListening = useCallback(() => {
    isCancelledRef.current = true
    isRecordingRef.current = false
    clearRestartTimer()

    const inst = recognitionRef.current
    recognitionRef.current = null
    cleanupRecognitionInstance(inst)

    setInterimTranscript('')
    setStatus('idle')
    onCancel?.()
  }, [clearRestartTimer, onCancel])

  const stopListening = useCallback(() => {
    isRecordingRef.current = false
    clearRestartTimer()

    if (recognitionRef.current) {
      setStatus('processing')
      try {
        recognitionRef.current.stop()
      } catch (e) {
        setStatus('idle')
      }
    } else {
      setStatus('idle')
    }
  }, [clearRestartTimer])

  const startRecognitionSession = useCallback(() => {
    const SpeechRecognitionClass = getSpeechRecognitionClass()
    if (!SpeechRecognitionClass) return null

    try {
      const recognition = new SpeechRecognitionClass()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = language || 'en-US'
      recognition.maxAlternatives = 1

      recognition.onstart = () => {
        if (!isCancelledRef.current && isRecordingRef.current) {
          setStatus('recording')
        }
      }

      recognition.onresult = (event: any) => {
        // If cancelled or no longer recording, ignore any trailing results
        if (isCancelledRef.current || !isRecordingRef.current) return

        let interim = ''
        let finalSegment = ''

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i]
          const text = result[0]?.transcript || ''
          if (result.isFinal) {
            finalSegment += text
          } else {
            interim += text
          }
        }

        if (finalSegment) {
          accumulatedTranscriptRef.current = (
            accumulatedTranscriptRef.current +
            (accumulatedTranscriptRef.current ? ' ' : '') +
            finalSegment
          ).trim()
          setTranscript(accumulatedTranscriptRef.current)
          onTranscript?.(accumulatedTranscriptRef.current, true)
        }

        setInterimTranscript(interim)
        if (interim) {
          onTranscript?.(interim, false)
        }
      }

      recognition.onerror = (event: any) => {
        if (isCancelledRef.current) return

        // Non-fatal transient silence in Chrome/Edge: do NOT terminate or show error while recording
        if (event.error === 'no-speech') {
          return
        }

        if (event.error === 'aborted') {
          if (!isRecordingRef.current) {
            setStatus('idle')
          }
          return
        }

        isRecordingRef.current = false
        clearRestartTimer()

        let message = 'Speech recognition failed. Please try again.'
        switch (event.error) {
          case 'not-allowed':
          case 'permission-denied':
            message = 'Microphone permission denied. Please allow microphone access in your browser settings.'
            break
          case 'audio-capture':
            message = 'No microphone was found. Please ensure a working microphone is connected.'
            break
          case 'network':
            message = 'Network error occurred during speech recognition. Please check your connection.'
            break
          default:
            message = `Speech recognition error: ${event.error || 'unknown'}. Please try again.`
        }

        setErrorMessage(message)
        setStatus('error')
        onError?.(message)
      }

      recognition.onend = () => {
        setInterimTranscript('')

        // Chrome/Edge closes its speech stream chunk every ~10-15 seconds.
        // Instead of calling .start() on the dead instance (which throws InvalidStateError in Chromium),
        // we cleanly clean up and instantiate a fresh session after 100ms so the user can speak continuously.
        if (isRecordingRef.current && !isCancelledRef.current) {
          cleanupRecognitionInstance(recognition)
          clearRestartTimer()
          restartTimerRef.current = setTimeout(() => {
            if (isRecordingRef.current && !isCancelledRef.current) {
              const nextSession = startRecognitionSession()
              recognitionRef.current = nextSession
            }
          }, 100)
        } else {
          isRecordingRef.current = false
          clearRestartTimer()
          setStatus('idle')
        }
      }

      recognition.start()
      return recognition
    } catch (err: any) {
      if (isRecordingRef.current && !isCancelledRef.current) {
        clearRestartTimer()
        restartTimerRef.current = setTimeout(() => {
          if (isRecordingRef.current && !isCancelledRef.current) {
            recognitionRef.current = startRecognitionSession()
          }
        }, 200)
      }
      return null
    }
  }, [clearRestartTimer, language, onError, onTranscript])

  const startListening = useCallback(() => {
    const SpeechRecognitionClass = getSpeechRecognitionClass()
    if (!SpeechRecognitionClass) {
      const err = 'Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.'
      setErrorMessage(err)
      setStatus('error')
      onError?.(err)
      return
    }

    if (isRecordingRef.current || status === 'recording' || status === 'processing') {
      stopListening()
      return
    }

    clearError()
    clearRestartTimer()
    isCancelledRef.current = false
    isRecordingRef.current = true
    accumulatedTranscriptRef.current = ''
    setTranscript('')
    setInterimTranscript('')

    const session = startRecognitionSession()
    if (!session) {
      isRecordingRef.current = false
      const message = 'Failed to start speech recognition. Please try again.'
      setErrorMessage(message)
      setStatus('error')
      onError?.(message)
      return
    }

    recognitionRef.current = session
  }, [clearError, clearRestartTimer, onError, startRecognitionSession, status, stopListening])

  return {
    isSupported,
    status,
    transcript,
    interimTranscript,
    errorMessage,
    language,
    supportedLanguages: SUPPORTED_LANGUAGES,
    startListening,
    stopListening,
    cancelListening,
    setLanguage,
    clearError,
  }
}
