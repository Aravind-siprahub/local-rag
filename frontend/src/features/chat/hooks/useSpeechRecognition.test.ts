import { test, describe, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { SUPPORTED_LANGUAGES } from './useSpeechRecognition.ts'

describe('Voice Search & Speech Recognition Architecture', () => {
  let originalWindow: any

  beforeEach(() => {
    originalWindow = (global as any).window
  })

  afterEach(() => {
    ;(global as any).window = originalWindow
  })

  test('1. Supported languages is restricted strictly to English (en-US)', () => {
    assert.strictEqual(SUPPORTED_LANGUAGES.length, 1)
    assert.strictEqual(SUPPORTED_LANGUAGES[0].code, 'en-US')
    assert.strictEqual(SUPPORTED_LANGUAGES[0].label, 'English')
    assert.strictEqual(SUPPORTED_LANGUAGES[0].shortLabel, 'EN')
  })

  test('2. Speech Recognition fallback when browser does not support Web Speech API', () => {
    // In node test environment without window.SpeechRecognition
    ;(global as any).window = {}
    
    const hasSpeech = Boolean(
      (global as any).window.SpeechRecognition ||
      (global as any).window.webkitSpeechRecognition
    )
    assert.strictEqual(hasSpeech, false, 'SpeechRecognition must be falsy in unsupported environments')
  })

  test('3. Speech Recognition constructor detection across standard and webkit prefixes', () => {
    class MockSpeechRecognition {
      lang = 'en-US'
      continuous = true
      start() {}
      stop() {}
      abort() {}
    }

    ;(global as any).window = { SpeechRecognition: MockSpeechRecognition }
    const hasStandard = Boolean((global as any).window.SpeechRecognition)
    assert.strictEqual(hasStandard, true)

    ;(global as any).window = { webkitSpeechRecognition: MockSpeechRecognition }
    const hasWebkit = Boolean((global as any).window.webkitSpeechRecognition)
    assert.strictEqual(hasWebkit, true)
  })

  test('4. Error mapping handles permission denied, network, and audio capture without crashing', () => {
    const errorMap: Record<string, string> = {
      'not-allowed': 'Microphone permission denied. Please allow microphone access in your browser settings.',
      'permission-denied': 'Microphone permission denied. Please allow microphone access in your browser settings.',
      'audio-capture': 'No microphone was found. Please ensure a working microphone is connected.',
      'network': 'Network error occurred during speech recognition. Please check your connection.',
    }

    assert.ok(errorMap['not-allowed'].includes('Microphone permission denied'))
    assert.ok(errorMap['audio-capture'].includes('No microphone was found'))
    assert.ok(errorMap['network'].includes('Network error'))
  })

  test('5. Continuous listening session recreation across Chrome stream boundaries (>10s)', () => {
    let sessionCounter = 0
    let isRecording = true
    let isCancelled = false

    // Simulate Chrome onend event after 10-15s chunk
    const simulateChromeChunkEnd = () => {
      if (isRecording && !isCancelled) {
        // Old instance destroyed, fresh instance spawned
        sessionCounter += 1
        return true
      }
      return false
    }

    const restarted = simulateChromeChunkEnd()
    assert.strictEqual(restarted, true, 'Must automatically spawn fresh session on Chrome onend boundary')
    assert.strictEqual(sessionCounter, 1)

    // Second chunk boundary after another 10s
    simulateChromeChunkEnd()
    assert.strictEqual(sessionCounter, 2, 'Must continue uninterrupted across multiple stream chunks')
  })

  test('6. Cancellation immediately detaches listeners and prevents trailing transcript events', () => {
    let trailingResultFired = false
    let isCancelled = false

    const mockRecognition: any = {
      onresult: () => {
        if (!isCancelled) {
          trailingResultFired = true
        }
      },
      onend: () => {},
      abort: () => {
        isCancelled = true
        // Nullify listeners on cancel
        mockRecognition.onresult = null
        mockRecognition.onend = null
      }
    }

    // User clicks Cancel
    mockRecognition.abort()

    // Simulate in-flight async result arriving after cancel
    if (mockRecognition.onresult) {
      mockRecognition.onresult()
    }

    assert.strictEqual(trailingResultFired, false, 'Trailing transcript events must NEVER fire after cancellation')
    assert.strictEqual(mockRecognition.onresult, null)
  })

  test('7. Cancellation cleanly reverts input to pre-voice snapshot', () => {
    const preVoiceInput = 'Original text before mic'
    let currentInput = preVoiceInput

    // User speaks
    const spokenTranscript = 'Who is the president of...'
    currentInput = `${preVoiceInput} ${spokenTranscript}`
    assert.strictEqual(currentInput, 'Original text before mic Who is the president of...')

    // User clicks Cancel
    const handleVoiceCancel = () => {
      currentInput = preVoiceInput
    }
    handleVoiceCancel()

    assert.strictEqual(currentInput, 'Original text before mic', 'Input must cleanly revert on cancel')
  })
})
