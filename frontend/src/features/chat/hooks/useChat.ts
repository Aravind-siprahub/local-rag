import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { chatService } from '../services/chat.service'
import type { ChatRequest } from '../types/chat'
import { useCurrentUser } from '@/hooks/useCurrentUser'

export const chatKeys = {
  all: ['chat'] as const,
  conversations: () => [...chatKeys.all, 'conversations'] as const,
  conversation: (id: string) => [...chatKeys.all, 'conversation', id] as const,
  messages: (id: string) => [...chatKeys.all, 'messages', id] as const,
}

export function useChat() {
  const queryClient = useQueryClient()
  const { data: user } = useCurrentUser()
  const userId = user?.id
  
  const { data: conversations, isLoading: isLoadingConversations } = useQuery({
    queryKey: chatKeys.conversations(),
    queryFn: () => chatService.listConversations(userId, { limit: 100 }),
    enabled: !!userId,
  })

  const createConversation = useMutation({
    mutationFn: (title?: string) => chatService.createConversation(userId || '', title),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: chatKeys.conversations() })
    },
  })

  const deleteConversation = useMutation({
    mutationFn: (id: string) => chatService.deleteConversation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: chatKeys.conversations() })
    },
  })

  const sendMessage = useMutation({
    mutationFn: (payload: ChatRequest) => chatService.sendMessage(payload),
    onSuccess: (_, variables) => {
      if (variables.session_id) {
        queryClient.invalidateQueries({ queryKey: chatKeys.messages(variables.session_id) })
      }
      queryClient.invalidateQueries({ queryKey: chatKeys.conversations() })
    },
  })

  return {
    conversations: conversations?.items || [],
    isLoadingConversations,
    createConversation,
    deleteConversation,
    /** @deprecated Prefer chatRequestStore + chatService.sendMessage for per-conversation UI state. */
    sendMessage,
  }
}

export function useConversationMessages(sessionId?: string) {
  return useQuery({
    queryKey: chatKeys.messages(sessionId!),
    queryFn: () => chatService.getConversationMessages(sessionId!, { limit: 100 }),
    enabled: !!sessionId,
  })
}
