import assert from 'node:assert/strict'
import { describe, it, beforeEach } from 'node:test'

import {
  DRAFT_CONVERSATION_KEY,
  chatRequestStore,
  type ConversationRequestState,
} from './chatRequest.store.js'

function getState(id: string): ConversationRequestState {
  return chatRequestStore.getConversation(id)
}

describe('chatRequestStore', () => {
  beforeEach(() => {
    chatRequestStore.reset()
  })

  it('keeps conversations independent: A loading does not lock B', () => {
    chatRequestStore.beginRequest('A', 'req-1')
    assert.equal(getState('A').status, 'loading')
    assert.equal(getState('B').status, 'idle')
    assert.equal(chatRequestStore.isSending('B'), false)
    assert.equal(chatRequestStore.isSending('A'), true)
  })

  it('rejects duplicate beginRequest for the same conversation', () => {
    const first = chatRequestStore.beginRequest('A', 'req-1')
    const second = chatRequestStore.beginRequest('A', 'req-2')
    assert.equal(first, true)
    assert.equal(second, false)
    assert.equal(getState('A').requestId, 'req-1')
  })

  it('always leaves loading on success', () => {
    chatRequestStore.beginRequest('A', 'req-1')
    chatRequestStore.completeRequest('A', 'req-1', 'success')
    assert.equal(getState('A').status, 'success')
    assert.equal(chatRequestStore.isSending('A'), false)
  })

  it('always leaves loading on error', () => {
    chatRequestStore.beginRequest('A', 'req-1')
    chatRequestStore.completeRequest('A', 'req-1', 'error', 'Failed')
    assert.equal(getState('A').status, 'error')
    assert.equal(getState('A').errorMessage, 'Failed')
    assert.equal(chatRequestStore.isSending('A'), false)
  })

  it('always leaves loading on timeout', () => {
    chatRequestStore.beginRequest('A', 'req-1')
    chatRequestStore.completeRequest('A', 'req-1', 'timeout', 'Response timed out. Please try again.')
    assert.equal(getState('A').status, 'timeout')
    assert.equal(chatRequestStore.isSending('A'), false)
  })

  it('always leaves loading on cancel', () => {
    chatRequestStore.beginRequest('A', 'req-1')
    chatRequestStore.completeRequest('A', 'req-1', 'cancelled')
    assert.equal(getState('A').status, 'cancelled')
    assert.equal(chatRequestStore.isSending('A'), false)
  })

  it('ignores stale completions from older request ids', () => {
    chatRequestStore.beginRequest('A', 'req-1')
    chatRequestStore.completeRequest('A', 'req-old', 'success')
    assert.equal(getState('A').status, 'loading')
    chatRequestStore.completeRequest('A', 'req-1', 'success')
    assert.equal(getState('A').status, 'success')
  })

  it('preserves request state when switching active conversation key', () => {
    chatRequestStore.beginRequest('A', 'req-1')
    chatRequestStore.setActiveConversation('B')
    assert.equal(chatRequestStore.getActiveConversationId(), 'B')
    assert.equal(getState('A').status, 'loading')
  })

  it('rekeys draft conversation state to a real session id', () => {
    chatRequestStore.beginRequest(DRAFT_CONVERSATION_KEY, 'req-1')
    chatRequestStore.rekeyConversation(DRAFT_CONVERSATION_KEY, 'session-123')
    assert.equal(getState(DRAFT_CONVERSATION_KEY).status, 'idle')
    assert.equal(getState('session-123').status, 'loading')
    assert.equal(getState('session-123').requestId, 'req-1')
  })

  it('runRequest guarantees cleanup after a long-running success', async () => {
    const started = Date.now()
    await chatRequestStore.runRequest('A', async () => {
      await new Promise((r) => setTimeout(r, 50))
      return 'ok'
    })
    assert.ok(Date.now() - started >= 50)
    assert.equal(chatRequestStore.isSending('A'), false)
    assert.equal(getState('A').status, 'success')
  })

  it('runRequest rekey moves loading state and still clears on completion', async () => {
    await chatRequestStore.runRequest(DRAFT_CONVERSATION_KEY, async (_signal: any, ctx: any) => {
      ctx.rekeyTo('session-xyz')
      assert.equal(chatRequestStore.isSending(DRAFT_CONVERSATION_KEY), false)
      assert.equal(chatRequestStore.isSending('session-xyz'), true)
      return 'ok'
    })
    assert.equal(chatRequestStore.isSending('session-xyz'), false)
    assert.equal(getState('session-xyz').status, 'success')
  })

  it('runRequest marks timeout and clears loading', async () => {
    await assert.rejects(
      () =>
        chatRequestStore.runRequest(
          'A',
          async (signal: any) => {
            await new Promise<void>((resolve, reject) => {
              const timer = setTimeout(() => resolve(), 5_000)
              signal.addEventListener('abort', () => {
                clearTimeout(timer)
                reject(Object.assign(new Error('aborted'), { code: 'ERR_CANCELED', name: 'CanceledError' }))
              })
            })
            return 'never'
          },
          { timeoutMs: 30 },
        ),
    )
    assert.equal(chatRequestStore.isSending('A'), false)
    assert.equal(getState('A').status, 'timeout')
  })

  it('runRequest prevents duplicate in-flight sends without blocking other conversations', async () => {
    const slow = chatRequestStore.runRequest('A', async () => {
      await new Promise((r) => setTimeout(r, 80))
      return 'a'
    })

    let blocked = false
    try {
      await chatRequestStore.runRequest('A', async () => 'dup')
    } catch {
      blocked = true
    }

    const b = await chatRequestStore.runRequest('B', async () => 'b')
    assert.equal(blocked, true)
    assert.equal(b, 'b')
    assert.equal(await slow, 'a')
    assert.equal(chatRequestStore.isSending('A'), false)
    assert.equal(chatRequestStore.isSending('B'), false)
  })

  it(
    'simulates a 2+ minute request without locking other conversations',
    { timeout: 180_000 },
    async () => {
      // Real wall-clock simulation of a long LLM generation (2m 10s).
      const TWO_PLUS_MINUTES_MS = 130_000
      const longRunning = chatRequestStore.runRequest('chat-a', async () => {
        await new Promise((r) => setTimeout(r, TWO_PLUS_MINUTES_MS))
        return 'done'
      })

      assert.equal(chatRequestStore.isSending('chat-a'), true)
      assert.equal(chatRequestStore.isSending('chat-b'), false)

      chatRequestStore.setActiveConversation('chat-b')
      assert.equal(chatRequestStore.getActiveConversationId(), 'chat-b')
      assert.equal(chatRequestStore.isSending('chat-a'), true)

      const bResult = await chatRequestStore.runRequest('chat-b', async () => {
        await new Promise((r) => setTimeout(r, 20))
        return 'b-ok'
      })
      assert.equal(bResult, 'b-ok')
      assert.equal(chatRequestStore.isSending('chat-b'), false)
      assert.equal(chatRequestStore.isSending('chat-a'), true)

      const result = await longRunning
      assert.equal(result, 'done')
      assert.equal(chatRequestStore.isSending('chat-a'), false)
      assert.equal(getState('chat-a').status, 'success')
    },
  )
})
