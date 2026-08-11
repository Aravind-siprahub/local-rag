import { chatRequestStore } from './chatRequest.store'

async function main() {
  chatRequestStore.reset()
  const t0 = Date.now()
  console.log('[sim] starting 130s request on chat-a')

  const longRunning = chatRequestStore.runRequest('chat-a', async () => {
    await new Promise((r) => setTimeout(r, 130_000))
    return 'done'
  })

  console.log('[sim] chat-a sending?', chatRequestStore.isSending('chat-a'))
  console.log('[sim] chat-b sending?', chatRequestStore.isSending('chat-b'))

  chatRequestStore.setActiveConversation('chat-b')
  console.log('[sim] switched active to chat-b')

  const b = await chatRequestStore.runRequest('chat-b', async () => {
    await new Promise((r) => setTimeout(r, 25))
    return 'b-ok'
  })
  console.log('[sim] chat-b completed:', b)
  console.log('[sim] chat-a still sending?', chatRequestStore.isSending('chat-a'))

  const a = await longRunning
  const elapsed = Date.now() - t0
  console.log('[sim] chat-a completed:', a)
  console.log('[sim] elapsedMs:', elapsed)
  console.log('[sim] chat-a sending after?', chatRequestStore.isSending('chat-a'))
  console.log('[sim] chat-a status:', chatRequestStore.getConversation('chat-a').status)

  if (elapsed < 130_000) {
    throw new Error(`Expected >= 130000ms, got ${elapsed}`)
  }
  if (chatRequestStore.isSending('chat-a')) {
    throw new Error('chat-a still sending after completion')
  }
  console.log('[sim] PASS')
}

main().catch((err) => {
  console.error('[sim] FAIL', err)
  process.exit(1)
})
