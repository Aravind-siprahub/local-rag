import axios from 'axios'
import { useState, useEffect } from 'react'
import { Menu, Info } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChat, useConversationMessages, chatKeys } from '@/features/chat/hooks/useChat'
import { useConversationRequestState } from '@/features/chat/hooks/useChatRequestStore'
import { chatService } from '@/features/chat/services/chat.service'
import {
  DRAFT_CONVERSATION_KEY,
  chatRequestStore,
} from '@/features/chat/store/chatRequest.store'
import { ChatSidebar } from '@/features/chat/components/ChatSidebar'
import { ChatHistory } from '@/features/chat/components/ChatHistory'
import { ChatInput } from '@/features/chat/components/ChatInput'
import type { Message, Citation } from '@/features/chat/types/chat'
import { getApiErrorMessage } from '@/api/client'
import { useQueryClient } from '@tanstack/react-query'

/** Per-conversation optimistic / overlay message state (survives navigation). */
const conversationOverlays = new Map<
  string,
  {
    messages: Message[]
    latestCitations?: Citation[]
    errorMessage?: { text: string; lastContent?: string } | null
  }
>()

const overlayListeners = new Set<() => void>()
function notifyOverlays() {
  overlayListeners.forEach((l) => l())
}

function getOverlay(id: string) {
  return conversationOverlays.get(id) ?? { messages: [], latestCitations: undefined, errorMessage: null }
}

function setOverlay(
  id: string,
  next: {
    messages: Message[]
    latestCitations?: Citation[]
    errorMessage?: { text: string; lastContent?: string } | null
  },
) {
  conversationOverlays.set(id, next)
  notifyOverlays()
}

export function ChatPage() {
  const [activeSessionId, setActiveSessionId] = useState<string | undefined>()
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
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

  // Sync server messages into the active conversation overlay when not mid-flight
  // with only optimistic temps — merge server + keep in-flight optimistic user msg.
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
        // Keep citations from the last in-memory reply if present
        latestCitations: existing.latestCitations,
      })
    }
  }, [messagesData, activeSessionId])

  const localMessages = overlay.messages
  const latestCitations = overlay.latestCitations
  const errorMessage = overlay.errorMessage

  const handleNewChat = () => {
    // Do NOT cancel the previous conversation's in-flight request.
    setActiveSessionId(undefined)
    if (!conversationOverlays.has(DRAFT_CONVERSATION_KEY)) {
      setOverlay(DRAFT_CONVERSATION_KEY, {
        messages: [],
        latestCitations: undefined,
        errorMessage: null,
      })
    } else {
      // Fresh draft each time New Chat is clicked if draft is idle
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

  const handleSend = async (content: string) => {
    const keyAtStart = activeSessionId ?? DRAFT_CONVERSATION_KEY
    if (chatRequestStore.isSending(keyAtStart)) return

    setErrorForActive(null)

    const tempUserMsg: Message = {
      id: `temp-${Date.now()}`,
      session_id: activeSessionId || 'temp',
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    }

    const before = getOverlay(keyAtStart)
    setOverlay(keyAtStart, {
      ...before,
      messages: [...before.messages, tempUserMsg],
      latestCitations: undefined,
      errorMessage: null,
    })

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
            const newSession = await createConversation.mutateAsync(title)
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

          const response = await chatService.sendMessage(
            {
              session_id: currentSessionId,
              question: content,
            },
            { signal, timeoutMs: 600_000 },
          )

          const assistantMsg: Message = {
            id: response.assistant_message_id,
            session_id: currentSessionId!,
            role: 'assistant',
            content: response.answer,
            model_used: response.model,
            total_tokens: response.token_usage?.total_tokens,
            created_at: new Date().toISOString(),
          }

          const current = getOverlay(workingKey)
          const withoutTemp = current.messages.filter((m) => m.id !== tempUserMsg.id)
          // Keep the user turn: replace temp with a stable local user message if server sync lags
          const userMsg: Message = {
            ...tempUserMsg,
            id: response.user_message_id || tempUserMsg.id,
            session_id: currentSessionId!,
          }
          setOverlay(workingKey, {
            ...current,
            messages: [...withoutTemp.filter((m) => m.id !== userMsg.id), userMsg, assistantMsg],
            latestCitations: response.citations,
            errorMessage: null,
          })

          void queryClient.invalidateQueries({ queryKey: chatKeys.messages(currentSessionId!) })
          void queryClient.invalidateQueries({ queryKey: chatKeys.conversations() })

          return response
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
        axios.isAxiosError(err) && (err.code === 'ERR_CANCELED' || err.name === 'CanceledError')

      let text = 'Failed to generate answer. Please try again.'
      if (isTimeout) {
        text = 'Response timed out. Please try again.'
      } else if (isCancelled) {
        text = 'Request was cancelled.'
      } else {
        const status = (err as { response?: { status?: number } })?.response?.status
        if (status === 503) {
          text =
            'AI model unavailable. Please check that Ollama is running locally and model is loaded.'
        } else {
          text = getApiErrorMessage(err) || text
        }
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

      <div className="flex-1 flex flex-col min-w-0 min-h-0 h-full relative z-0">
        <div className="h-14 border-b flex items-center px-4 bg-background z-10 shrink-0">
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden mr-2"
            onClick={() => setIsMobileSidebarOpen(true)}
          >
            <Menu className="h-5 w-5" />
          </Button>
          <div className="flex-1 min-w-0">
            <h1 className="text-lg font-semibold truncate">
              {activeConversation ? activeConversation.title : 'New Chat'}
            </h1>
          </div>
          <div className="flex items-center text-xs text-muted-foreground ml-2 shrink-0 gap-1 bg-muted/50 px-2 py-1 rounded-md">
            <Info className="w-3 h-3" />
            <span className="hidden sm:inline">Local RAG mode</span>
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
            loadingLabel={showGenerating ? 'Generating response...' : undefined}
          />
        </div>

        <div className="p-4 bg-background border-t shrink-0">
          <div className="max-w-3xl mx-auto">
            <ChatInput
              onSend={(msg) => {
                void handleSend(msg)
              }}
              sendDisabled={currentMessageSending}
              disabled={false}
            />
            <p className="text-center text-xs text-muted-foreground mt-2">
              Responses are generated by a local AI model and may contain inaccuracies.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
