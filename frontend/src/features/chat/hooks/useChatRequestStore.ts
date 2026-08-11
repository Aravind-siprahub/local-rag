import { useSyncExternalStore } from 'react'

import {
  chatRequestStore,
  type ConversationRequestState,
} from '../store/chatRequest.store'

const IDLE_STATE: ConversationRequestState = {
  status: 'idle',
  requestId: null,
  abortController: null,
}

export function useConversationSending(conversationId: string | undefined): boolean {
  return useSyncExternalStore(
    chatRequestStore.subscribe,
    () => (conversationId ? chatRequestStore.isSending(conversationId) : false),
    () => (conversationId ? chatRequestStore.isSending(conversationId) : false),
  )
}

export function useConversationRequestState(
  conversationId: string | undefined,
): ConversationRequestState {
  return useSyncExternalStore(
    chatRequestStore.subscribe,
    () => (conversationId ? chatRequestStore.getConversation(conversationId) : IDLE_STATE),
    () => (conversationId ? chatRequestStore.getConversation(conversationId) : IDLE_STATE),
  )
}
