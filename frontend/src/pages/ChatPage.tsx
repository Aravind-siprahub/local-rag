import { useState, useEffect } from 'react'
import { Menu } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import { ChatSidebar } from '@/features/chat/components/ChatSidebar'
import { ChatHistory } from '@/features/chat/components/ChatHistory'
import { ChatInput } from '@/features/chat/components/ChatInput'

import { useChat, useConversationMessages, chatKeys } from '@/features/chat/hooks/useChat'
import {
  chatRequestStore,
  DRAFT_CONVERSATION_KEY,
} from '@/features/chat/store/chatRequest.store'
import { useConversationRequestState } from '@/features/chat/hooks/useChatRequestStore'
import type { Message, Citation } from '@/features/chat/types/chat'
import { chatService } from '@/features/chat/services/chat.service'
import { Button } from '@/components/ui/button'

interface ConversationOverlayState {
  messages: Message[]
  latestCitations?: Citation[]
  errorMessage?: { text: string; lastContent?: string } | null
  streamingStatus?: string
}

const conversationOverlays = new Map<string, ConversationOverlayState>()
const overlayListeners = new Set<() => void>()

function notifyOverlays() {
  overlayListeners.forEach((l) => l())
}

function getOverlay(id: string): ConversationOverlayState {
  return (
    conversationOverlays.get(id) ?? {
      messages: [],
      latestCitations: undefined,
      errorMessage: null,
    }
  )
}

function setOverlay(id: string, state: ConversationOverlayState) {
  conversationOverlays.set(id, state)
  notifyOverlays()
}

export function ChatPage() {
  const [activeSessionId, setActiveSessionId] = useState<string | undefined>()
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState<string>('qwen3:8b')
  const [editingMessage, setEditingMessage] = useState<Message | null>(null)
  const [, setOverlayVersion] = useState(0)
  const queryClient = useQueryClient()

  useEffect(() => {
    const listener = () => setOverlayVersion((v) => v + 1)
    overlayListeners.add(listener)
    return () => {
      overlayListeners.delete(listener)
    }
  }, [])

  const {
    conversations,
    createConversation,
    deleteConversation,
  } = useChat()

  const conversationKey = activeSessionId ?? DRAFT_CONVERSATION_KEY
  const requestState = useConversationRequestState(
    activeSessionId ? activeSessionId : DRAFT_CONVERSATION_KEY,
  )
  const currentMessageSending = requestState.status === 'loading'
  const { data: messagesData, isLoading: currentConversationLoading } =
    useConversationMessages(activeSessionId)

  const overlay = getOverlay(conversationKey)

  // Sync server messages into active conversation overlay when not mid-flight
  useEffect(() => {
    chatRequestStore.setActiveConversation(activeSessionId)

    if (!activeSessionId) {
      return
    }

    if (!messagesData?.items) {
      return
    }

    const sorted = [...messagesData.items].sort(
      (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    )

    const existing = getOverlay(activeSessionId)
    const optimisticTemps = existing.messages.filter((m) => m.id.startsWith('temp-'))
    const isSending = chatRequestStore.isSending(activeSessionId)

    if (isSending && optimisticTemps.length > 0) {
      const serverIds = new Set(sorted.map((m) => m.id))
      const stillMissing = optimisticTemps.filter((m) => !serverIds.has(m.id))
      setOverlay(activeSessionId, {
        ...existing,
        messages: [...sorted, ...stillMissing],
      })
    } else if (!isSending) {
      setOverlay(activeSessionId, {
        ...existing,
        messages: sorted,
        latestCitations: existing.latestCitations,
      })
    }
  }, [messagesData, activeSessionId])

  const localMessages = overlay.messages
  const latestCitations = overlay.latestCitations
  const errorMessage = overlay.errorMessage

  const handleNewChat = () => {
    setActiveSessionId(undefined)
    setEditingMessage(null)
    if (!conversationOverlays.has(DRAFT_CONVERSATION_KEY)) {
      setOverlay(DRAFT_CONVERSATION_KEY, {
        messages: [],
        latestCitations: undefined,
        errorMessage: null,
      })
    } else {
      if (!chatRequestStore.isSending(DRAFT_CONVERSATION_KEY)) {
        setOverlay(DRAFT_CONVERSATION_KEY, {
          messages: [],
          latestCitations: undefined,
          errorMessage: null,
        })
      }
    }
  }

  const handleSelectConversation = (id: string) => {
    setActiveSessionId(id)
    setEditingMessage(null)
  }

  const handleStop = () => {
    chatRequestStore.abortRequest(activeSessionId ?? DRAFT_CONVERSATION_KEY)
  }

  const handleDeleteChat = async (id: string) => {
    await deleteConversation.mutateAsync(id)
    conversationOverlays.delete(id)
    if (activeSessionId === id) {
      handleNewChat()
    }
  }

  const setErrorForActive = (value: { text: string; lastContent?: string } | null) => {
    const current = getOverlay(conversationKey)
    setOverlay(conversationKey, { ...current, errorMessage: value })
  }

  const handleEditUserMessage = (msg: Message) => {
    if (msg.role === 'user') {
      setEditingMessage(msg)
    }
  }

  const handleRegenerateAssistantMessage = (assistantMsg: Message) => {
    if (assistantMsg.role !== 'assistant') return
    const msgs = getOverlay(conversationKey).messages
    const idx = msgs.findIndex((m) => m.id === assistantMsg.id)
    if (idx > 0 && msgs[idx - 1]?.role === 'user') {
      const userMsg = msgs[idx - 1]
      // Backend does not support re-submitting images by ID/URL.
      // We explicitly drop the localImageUrl to avoid falsely showing an image in the new user bubble that the backend never received.
      void handleSend(userMsg.content, undefined, undefined)
    }
  }

  const handleSend = async (content: string, file?: File, preservedImageUrl?: string) => {
    const keyAtStart = activeSessionId ?? DRAFT_CONVERSATION_KEY
    if (chatRequestStore.isSending(keyAtStart)) return

    setErrorForActive(null)

    const resolvedImageUrl = file
      ? URL.createObjectURL(file)
      : preservedImageUrl || undefined

    const tempUserMsg: Message = {
      id: `temp-${Date.now()}`,
      session_id: activeSessionId || 'temp',
      role: 'user',
      content,
      created_at: new Date().toISOString(),
      localImageUrl: resolvedImageUrl,
    }

    const before = getOverlay(keyAtStart)

    // If editing a historical user message, truncate messages from that point onward
    let baseMessages = before.messages
    if (editingMessage) {
      const editIdx = baseMessages.findIndex((m) => m.id === editingMessage.id)
      if (editIdx >= 0) {
        baseMessages = baseMessages.slice(0, editIdx)
      }
    }

    setOverlay(keyAtStart, {
      ...before,
      messages: [...baseMessages, tempUserMsg],
      latestCitations: undefined,
      errorMessage: null,
    })

    setEditingMessage(null)

    try {
      await chatRequestStore.runRequest(
        keyAtStart,
        async (signal, ctx) => {
          let currentSessionId = activeSessionId
          let workingKey = keyAtStart

          if (!currentSessionId) {
            const title =
              content.split(' ').slice(0, 5).join(' ') +
              (content.split(' ').length > 5 ? '...' : '')
            const newSession = await createConversation.mutateAsync(
              title || (file || preservedImageUrl ? 'Image analysis' : 'New chat'),
            )
            currentSessionId = newSession.id
            ctx.rekeyTo(currentSessionId)
            const draftOverlay = getOverlay(DRAFT_CONVERSATION_KEY)
            setOverlay(currentSessionId, {
              ...draftOverlay,
              messages: draftOverlay.messages.map((m) =>
                m.id === tempUserMsg.id ? { ...m, session_id: currentSessionId! } : m,
              ),
            })
            setOverlay(DRAFT_CONVERSATION_KEY, {
              messages: [],
              latestCitations: undefined,
              errorMessage: null,
            })
            setActiveSessionId(newSession.id)
            workingKey = currentSessionId
          }

          let assistantMsg: Message = {
            id: `temp-assistant-${Date.now()}`,
            session_id: currentSessionId!,
            role: 'assistant',
            content: '',
            created_at: new Date().toISOString(),
          }

          const currentForStart = getOverlay(workingKey)
          const withoutTempForStart = currentForStart.messages.filter((m) => m.id !== tempUserMsg.id)
          setOverlay(workingKey, {
            ...currentForStart,
            messages: [...withoutTempForStart, tempUserMsg, assistantMsg],
            streamingStatus: 'Starting...',
          })

          await chatService.streamMessage(
            {
              session_id: currentSessionId,
              question: content,
              file,
            },
            {
              onStatus: (message) => {
                const current = getOverlay(workingKey)
                setOverlay(workingKey, { ...current, streamingStatus: message })
              },
              onMeta: (sources, userMessageId) => {
                const current = getOverlay(workingKey)
                const mappedMessages = current.messages.map((m) => {
                  if (m.id === tempUserMsg.id) return { ...m, id: userMessageId }
                  return m
                })
                setOverlay(workingKey, {
                  ...current,
                  messages: mappedMessages,
                  latestCitations: sources,
                })
              },
              onToken: (text) => {
                assistantMsg = { ...assistantMsg, content: assistantMsg.content + text }
                const current = getOverlay(workingKey)
                const mappedMessages = current.messages.map((m) =>
                  m.id === assistantMsg.id ? assistantMsg : m
                )
                setOverlay(workingKey, { ...current, messages: mappedMessages })
              },
              onDone: (data) => {
                assistantMsg = {
                  ...assistantMsg,
                  id: data.assistant_message_id || assistantMsg.id,
                  latency_ms: data.processing_time_ms,
                  ttft_ms: data.ttft_ms,
                  total_tokens: data.token_count,
                  model_used: data.model || 'ollama',
                }
                const current = getOverlay(workingKey)
                const mappedMessages = current.messages.map((m) =>
                  m.id === `temp-assistant-${Date.now()}` || m.role === 'assistant' && m.id === assistantMsg.id ? assistantMsg : m
                )
                setOverlay(workingKey, {
                  ...current,
                  messages: mappedMessages,
                  streamingStatus: undefined,
                })
              },
              onError: () => {
                // error handled by throw and catch in chatRequestStore/UI wrapper below
              }
            },
            { signal }
          )

          void queryClient.invalidateQueries({ queryKey: chatKeys.messages(currentSessionId!) })
          void queryClient.invalidateQueries({ queryKey: chatKeys.conversations() })
        },
        { timeoutMs: 600_000 },
      )
    } catch (err: unknown) {
      const current = getOverlay(activeSessionId ?? keyAtStart)
      const withoutTemp = current.messages.filter((m) => m.id !== tempUserMsg.id)

      const isTimeout =
        (err as { code?: string })?.code === 'TIMEOUT' ||
        (axios.isAxiosError(err) && err.code === 'ECONNABORTED')

      const isCancelled =
        (err instanceof Error && err.name === 'AbortError') ||
        (axios.isAxiosError(err) && (err.code === 'ERR_CANCELED' || err.name === 'CanceledError'))

      let text = 'Failed to generate answer. Please try again.'
      if (isTimeout) {
        text = 'Response timed out. Please try again.'
      } else if (isCancelled) {
        text = 'Request was cancelled.'
      }

      setOverlay(activeSessionId ?? keyAtStart, {
        ...current,
        messages: withoutTemp,
        errorMessage: { text, lastContent: content },
      })
    }
  }

  const activeConversation = conversations.find((c) => c.id === activeSessionId)
  const showGenerating = currentMessageSending

  return (
    <div className="flex h-full min-h-0 w-full bg-background overflow-hidden relative">
      <ChatSidebar
        conversations={conversations}
        activeId={activeSessionId}
        onSelect={handleSelectConversation}
        onNew={handleNewChat}
        onDelete={handleDeleteChat}
        isMobileOpen={isMobileSidebarOpen}
        setMobileOpen={setIsMobileSidebarOpen}
      />

      <div className="flex-1 flex flex-col min-w-0 min-h-0 h-full relative z-0 bg-background/50">
        <div className="h-13 border-b border-border/35 flex items-center justify-between px-4 bg-background/95 backdrop-blur-md z-10 shrink-0">
          <div className="flex items-center min-w-0 flex-1">
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden mr-2 shrink-0 h-8 w-8 text-muted-foreground hover:text-foreground"
              onClick={() => setIsMobileSidebarOpen(true)}
            >
              <Menu className="h-4 w-4" />
            </Button>
            <h1 className="text-sm font-semibold truncate text-foreground/90 font-display">
              {activeConversation ? activeConversation.title : 'New Chat'}
            </h1>
          </div>

          <div className="flex items-center gap-2 shrink-0 ml-2">
            <div className="hidden sm:flex items-center text-[10px] text-muted-foreground/80 gap-1 bg-muted/40 border border-border/20 px-2 py-0.5 rounded-md font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>Local RAG</span>
            </div>
          </div>
        </div>

        {errorMessage ? (
          <div className="mx-4 mt-3 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-xs flex items-center justify-between gap-3 shrink-0">
            <span>{errorMessage.text}</span>
            {errorMessage.lastContent ? (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-7 text-xs bg-background shrink-0"
                disabled={currentMessageSending}
                onClick={() => {
                  if (errorMessage.lastContent) {
                    void handleSend(errorMessage.lastContent)
                  }
                }}
              >
                Retry
              </Button>
            ) : null}
          </div>
        ) : null}

        <div className="flex-1 min-h-0 flex flex-col">
          <ChatHistory
            messages={localMessages}
            isLoading={showGenerating || (!!activeSessionId && currentConversationLoading && localMessages.length === 0)}
            latestCitations={latestCitations}
            loadingLabel={overlay.streamingStatus || (showGenerating ? 'Generating response...' : undefined)}
            onEditMessage={handleEditUserMessage}
            onRegenerateMessage={handleRegenerateAssistantMessage}
            onSuggestedClick={(text) => handleSend(text)}
          />
        </div>

        <div className="p-3 sm:p-4 bg-background/50 border-t border-border/30 shrink-0">
          <div className="max-w-4xl mx-auto">
            <ChatInput
              onSend={(msg, file, preservedImageUrl) => {
                void handleSend(msg, file, preservedImageUrl)
              }}
              onStop={handleStop}
              sendDisabled={currentMessageSending}
              disabled={false}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
              editingMessage={editingMessage}
              onCancelEdit={() => setEditingMessage(null)}
            />
            <p className="text-center text-[10px] text-muted-foreground/40 mt-2 leading-relaxed">
              Responses are generated by local AI models ({selectedModel}) and may contain inaccuracies.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
