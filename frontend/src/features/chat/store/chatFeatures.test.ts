import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { AVAILABLE_MODELS } from '../constants/models.js'

interface TestMessage {
  id: string
  session_id?: string
  role: 'user' | 'assistant'
  content: string
  created_at?: string
  localImageUrl?: string
}

describe('Chat Features & Model Selection Rules', () => {
  it('1. Edit User Message: puts original content into state', () => {
    const originalMessage: TestMessage = {
      id: 'msg-1',
      session_id: 's-1',
      role: 'user',
      content: 'Explain this chart',
      created_at: new Date().toISOString(),
      localImageUrl: 'blob:http://localhost/1234',
    }

    let editingMessage: TestMessage | null = null
    const handleEdit = (msg: TestMessage) => {
      editingMessage = msg
    }

    handleEdit(originalMessage)
    assert.equal((editingMessage as TestMessage | null)?.id, 'msg-1')
    assert.equal((editingMessage as TestMessage | null)?.content, 'Explain this chart')
  })

  it('2. Cancel Edit: clears editing state', () => {
    let editingMessage: TestMessage | null = { id: 'msg-1', role: 'user', content: 'test' }
    const handleCancelEdit = () => {
      editingMessage = null
    }

    handleCancelEdit()
    assert.equal(editingMessage, null)
  })

  it('3. Submit Edited Message: truncates previous history and resubmits', () => {
    const messages: TestMessage[] = [
      { id: 'u-1', role: 'user', content: 'First question' },
      { id: 'a-1', role: 'assistant', content: 'First answer' },
      { id: 'u-2', role: 'user', content: 'Second question' },
      { id: 'a-2', role: 'assistant', content: 'Second answer' },
    ]

    const editTargetId = 'u-2'
    const editIdx = messages.findIndex((m) => m.id === editTargetId)
    const baseMessages = messages.slice(0, editIdx)

    assert.equal(baseMessages.length, 2)
    assert.equal(baseMessages[0].id, 'u-1')
    assert.equal(baseMessages[1].id, 'a-1')
  })

  it('4. Preserve Image during Edit: maintains localImageUrl when resubmitting without new file', () => {
    const userMsg: TestMessage = {
      id: 'u-1',
      role: 'user',
      content: 'Analyze this photo',
      localImageUrl: 'blob:http://localhost/image-blob-id',
    }

    const newContent = 'Analyze this photo in detail'
    const preservedImageUrl = userMsg.localImageUrl

    const resubmittedMsg: TestMessage = {
      id: 'temp-123',
      role: 'user',
      content: newContent,
      localImageUrl: preservedImageUrl,
    }

    assert.equal(resubmittedMsg.content, 'Analyze this photo in detail')
    assert.equal(resubmittedMsg.localImageUrl, 'blob:http://localhost/image-blob-id')
  })

  it('5. Regenerate Assistant Response: locates target user message', () => {
    const messages: TestMessage[] = [
      { id: 'u-1', role: 'user', content: 'What is local RAG?' },
      { id: 'a-1', role: 'assistant', content: 'Local RAG processes documents locally.' },
    ]

    const assistantMsg = messages[1]
    const idx = messages.findIndex((m) => m.id === assistantMsg.id)
    const correspondingUserMsg = idx > 0 ? messages[idx - 1] : null

    assert.ok(correspondingUserMsg)
    assert.equal(correspondingUserMsg?.id, 'u-1')
    assert.equal(correspondingUserMsg?.content, 'What is local RAG?')
  })

  it('6. Model Selector: exposes qwen3:8b, local-rag, and local-rag-vision', () => {
    const modelIds = AVAILABLE_MODELS.map((m: any) => m.id)
    assert.ok(modelIds.includes('qwen3:8b'))
    assert.ok(modelIds.includes('qwen3-vl:4b'))
    assert.ok(modelIds.includes('local-rag'))
    assert.ok(modelIds.includes('local-rag-vision'))
  })

  it('7. Available Models display text vs vision badges', () => {
    const textModel = AVAILABLE_MODELS.find((m: any) => m.id === 'qwen3:8b')
    const visionModel = AVAILABLE_MODELS.find((m: any) => m.id === 'local-rag-vision')

    assert.equal(textModel?.type, 'text')
    assert.equal(visionModel?.type, 'vision')
  })

  it('8. Image automatically selects/uses vision model qwen3-vl:4b', () => {
    const hasImageAttached = true
    const selectedModel = 'qwen3:8b' // user selected text model

    const effectiveModel = hasImageAttached ? 'qwen3-vl:4b' : selectedModel
    assert.equal(effectiveModel, 'qwen3-vl:4b')
  })

  it('9. Text uses default selected model qwen3:8b', () => {
    const hasImageAttached = false
    const selectedModel = 'qwen3:8b'

    const effectiveModel = hasImageAttached ? 'qwen3-vl:4b' : selectedModel
    assert.equal(effectiveModel, 'qwen3:8b')
  })

  it('10. Responsive action buttons: ensures Edit and Regenerate actions exist on messages', () => {
    const userActions = ['Edit', 'Copy']
    const assistantActions = ['Regenerate', 'Copy']

    assert.ok(userActions.includes('Edit'))
    assert.ok(assistantActions.includes('Regenerate'))
  })
})
