export const DRAFT_CONVERSATION_KEY = '__draft__'

export type RequestLifecycleStatus =
  | 'idle'
  | 'loading'
  | 'success'
  | 'error'
  | 'cancelled'
  | 'timeout'

export interface ConversationRequestState {
  status: RequestLifecycleStatus
  requestId: string | null
  errorMessage?: string
  abortController: AbortController | null
}

const idleState = (): ConversationRequestState => ({
  status: 'idle',
  requestId: null,
  errorMessage: undefined,
  abortController: null,
})

let conversations = new Map<string, ConversationRequestState>()
let activeConversationId: string | undefined = undefined
const listeners = new Set<() => void>()

function notify() {
  listeners.forEach((listener) => listener())
}

const SHARED_IDLE: ConversationRequestState = {
  status: 'idle',
  requestId: null,
  errorMessage: undefined,
  abortController: null,
}

function ensure(id: string): ConversationRequestState {
  const existing = conversations.get(id)
  if (existing) return existing
  const created = idleState()
  conversations.set(id, created)
  return created
}

export const chatRequestStore = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener)
    return () => listeners.delete(listener)
  },

  getSnapshot(): Map<string, ConversationRequestState> {
    return conversations
  },

  getActiveConversationId(): string | undefined {
    return activeConversationId
  },

  setActiveConversation(id: string | undefined): void {
    activeConversationId = id
    notify()
  },

  getConversation(id: string): ConversationRequestState {
    return conversations.get(id) ?? SHARED_IDLE
  },

  isSending(id: string): boolean {
    return (conversations.get(id)?.status ?? 'idle') === 'loading'
  },

  beginRequest(conversationId: string, requestId: string): boolean {
    const current = ensure(conversationId)
    if (current.status === 'loading') {
      return false
    }
    conversations.set(conversationId, {
      status: 'loading',
      requestId,
      errorMessage: undefined,
      abortController: new AbortController(),
    })
    notify()
    return true
  },

  completeRequest(
    conversationId: string,
    requestId: string,
    status: Exclude<RequestLifecycleStatus, 'idle' | 'loading'>,
    errorMessage?: string,
  ): void {
    const current = ensure(conversationId)
    if (current.requestId !== requestId) {
      return
    }
    conversations.set(conversationId, {
      status,
      requestId: null,
      errorMessage,
      abortController: null,
    })
    notify()
  },

  rekeyConversation(fromId: string, toId: string): void {
    const current = conversations.get(fromId)
    if (!current) return
    conversations.set(toId, { ...current })
    conversations.set(fromId, idleState())
    if (activeConversationId === fromId) {
      activeConversationId = toId
    }
    notify()
  },

  reset(): void {
    conversations = new Map()
    activeConversationId = undefined
    notify()
  },

  abortRequest(conversationId: string): void {
    const current = conversations.get(conversationId)
    if (current && current.status === 'loading' && current.abortController) {
      current.abortController.abort()
      this.completeRequest(conversationId, current.requestId!, 'cancelled', 'Request stopped by user.')
    }
  },

  async runRequest<T>(
    conversationId: string,
    task: (
      signal: AbortSignal,
      ctx: { rekeyTo: (toId: string) => void; getConversationId: () => string },
    ) => Promise<T>,
    options?: { timeoutMs?: number },
  ): Promise<T> {
    const requestId = `req-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    const started = this.beginRequest(conversationId, requestId)
    if (!started) {
      throw new Error('DUPLICATE_REQUEST')
    }

    let currentId = conversationId
    const state = ensure(currentId)
    const controller = state.abortController ?? new AbortController()
    const timeoutMs = options?.timeoutMs
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined

    if (timeoutMs && timeoutMs > 0) {
      timeoutHandle = setTimeout(() => {
        controller.abort()
      }, timeoutMs)
    }

    const ctx = {
      getConversationId: () => currentId,
      rekeyTo: (toId: string) => {
        if (toId === currentId) return
        this.rekeyConversation(currentId, toId)
        currentId = toId
      },
    }

    try {
      const result = await task(controller.signal, ctx)
      if (controller.signal.aborted) {
        this.completeRequest(currentId, requestId, 'timeout', 'Response timed out. Please try again.')
        throw Object.assign(new Error('Response timed out. Please try again.'), { code: 'TIMEOUT' })
      }
      this.completeRequest(currentId, requestId, 'success')
      return result
    } catch (error) {
      const axiosCode = (error as { code?: string })?.code
      const isTimeoutCode = axiosCode === 'ECONNABORTED' || axiosCode === 'TIMEOUT'
      const isAbort =
        isTimeoutCode ||
        controller.signal.aborted ||
        (error instanceof Error &&
          (error.name === 'CanceledError' ||
            error.name === 'AbortError' ||
            (error as { code?: string }).code === 'ERR_CANCELED'))

      if (isAbort) {
        this.completeRequest(
          currentId,
          requestId,
          'timeout',
          'Response timed out. Please try again.',
        )
        throw Object.assign(new Error('Response timed out. Please try again.'), { code: 'TIMEOUT' })
      }

      const message = error instanceof Error ? error.message : 'Failed to generate answer. Please try again.'
      this.completeRequest(currentId, requestId, 'error', message)
      throw error
    } finally {
      if (timeoutHandle) clearTimeout(timeoutHandle)
      if (this.isSending(currentId) && ensure(currentId).requestId === requestId) {
        this.completeRequest(currentId, requestId, 'error', 'Request ended unexpectedly.')
      }
    }
  },
}
