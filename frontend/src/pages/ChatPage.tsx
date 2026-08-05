import { useState, useEffect } from 'react'
import { Menu, Info } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChat, useConversationMessages } from '@/features/chat/hooks/useChat'
import { ChatSidebar } from '@/features/chat/components/ChatSidebar'
import { ChatHistory } from '@/features/chat/components/ChatHistory'
import { ChatInput } from '@/features/chat/components/ChatInput'
import type { Message, Citation } from '@/features/chat/types/chat'

export function ChatPage() {
  const [activeSessionId, setActiveSessionId] = useState<string | undefined>()
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
  const [localMessages, setLocalMessages] = useState<Message[]>([])
  const [latestCitations, setLatestCitations] = useState<Citation[] | undefined>()

  const {
    conversations,
    createConversation,
    deleteConversation,
    sendMessage,
  } = useChat()

  const { data: messagesData, isLoading: isLoadingMessages } = useConversationMessages(activeSessionId)

  // Sync messages from backend
  useEffect(() => {
    if (messagesData?.items) {
      // API returns them descending or ascending? Usually descending, but we want ascending for chat UI
      // Let's assume the API returns them newest first (common pattern for offset pagination), so we reverse.
      // Wait, let's just sort by created_at.
      const sorted = [...messagesData.items].sort(
        (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      )
      setLocalMessages(sorted)
      // When we load a new conversation, clear latest citations since we don't store them in db yet
      setLatestCitations(undefined)
    } else {
      setLocalMessages([])
    }
  }, [messagesData, activeSessionId])

  const handleNewChat = () => {
    setActiveSessionId(undefined)
    setLocalMessages([])
    setLatestCitations(undefined)
  }

  const handleDeleteChat = async (id: string) => {
    await deleteConversation.mutateAsync(id)
    if (activeSessionId === id) {
      handleNewChat()
    }
  }

  const [errorMessage, setErrorMessage] = useState<{ text: string; lastContent?: string } | null>(null)

  const handleSend = async (content: string) => {
    setErrorMessage(null)
    const tempUserMsg: Message = {
      id: `temp-${Date.now()}`,
      session_id: activeSessionId || 'temp',
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    }
    setLocalMessages((prev) => [...prev, tempUserMsg])
    setLatestCitations(undefined)

    try {
      let currentSessionId = activeSessionId
      if (!currentSessionId) {
        const title =
          content.split(' ').slice(0, 5).join(' ') + (content.split(' ').length > 5 ? '...' : '')
        const newSession = await createConversation.mutateAsync(title)
        currentSessionId = newSession.id
        setActiveSessionId(newSession.id)
      }

      const response = await sendMessage.mutateAsync({
        session_id: currentSessionId,
        question: content,
      })

      const assistantMsg: Message = {
        id: response.assistant_message_id,
        session_id: currentSessionId,
        role: 'assistant',
        content: response.answer,
        model_used: response.model,
        total_tokens: response.token_usage?.total_tokens,
        created_at: new Date().toISOString(),
      }
      setLocalMessages((prev) => [...prev, assistantMsg])
      setLatestCitations(response.citations)
    } catch (err: unknown) {
      setLocalMessages((prev) => prev.filter((m) => m.id !== tempUserMsg.id))

      const status = (err as { response?: { status?: number } })?.response?.status
      if (status === 503) {
        setErrorMessage({
          text: 'AI model unavailable. Please check that Ollama is running locally and model is loaded.',
          lastContent: content,
        })
      } else {
        setErrorMessage({
          text: 'Failed to generate answer. Please try again.',
          lastContent: content,
        })
      }
    }
  }

  const activeConversation = conversations.find((c) => c.id === activeSessionId)

  return (
    <div className="flex h-[calc(100vh-4rem)] w-full bg-background overflow-hidden relative">
      <ChatSidebar
        conversations={conversations}
        activeId={activeSessionId}
        onSelect={setActiveSessionId}
        onNew={handleNewChat}
        onDelete={handleDeleteChat}
        isMobileOpen={isMobileSidebarOpen}
        setMobileOpen={setIsMobileSidebarOpen}
      />

      <div className="flex-1 flex flex-col min-w-0 h-full relative z-0">
        {/* Header */}
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

        {/* Error Notification Banner */}
        {errorMessage ? (
          <div className="mx-4 mt-3 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-xs flex items-center justify-between gap-3 shrink-0">
            <span>{errorMessage.text}</span>
            {errorMessage.lastContent ? (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-7 text-xs bg-background shrink-0"
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

        {/* Chat History */}
        <ChatHistory
          messages={localMessages}
          isLoading={isLoadingMessages || sendMessage.isPending}
          latestCitations={latestCitations}
        />

        {/* Input Area */}
        <div className="p-4 bg-background border-t shrink-0">
          <div className="max-w-3xl mx-auto">
            <ChatInput
              onSend={(msg) => {
                void handleSend(msg)
              }}
              disabled={sendMessage.isPending || isLoadingMessages}
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
