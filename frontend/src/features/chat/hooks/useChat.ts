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
    onMutate: async (id: string) => {
      await queryClient.cancelQueries({ queryKey: chatKeys.conversations() })
      const previous = queryClient.getQueryData<any>(chatKeys.conversations())

      if (previous) {
        if (Array.isArray(previous.items)) {
          queryClient.setQueryData(chatKeys.conversations(), {
            ...previous,
            items: previous.items.filter((c: any) => c.id !== id),
            total: Math.max(0, (previous.total || previous.items.length) - 1),
          })
        } else if (Array.isArray(previous)) {
          queryClient.setQueryData(
            chatKeys.conversations(),
            previous.filter((c: any) => c.id !== id)
          )
        }
      }

      return { previous }
    },
    onError: (_err, _id, context) => {
      if (context?.previous) {
        queryClient.setQueryData(chatKeys.conversations(), context.previous)
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: chatKeys.conversations() })
    },
  })

  const sendMessage = useMutation({
    mutationFn: async (payload: ChatRequest) => {
      console.log("[9] sendMessage mutation called", payload);
      try {
        const res = await chatService.sendMessage(payload)
        console.log("[10] sendMessage mutation completed");
        return res
      } catch (error) {
        console.error("[10A] Mutation failed", error);
        throw error
      }
    },
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
