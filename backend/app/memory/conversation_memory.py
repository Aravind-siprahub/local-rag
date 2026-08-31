"""Short-term conversation memory interface.

Wraps the existing ChatMessageService to provide a clean, bounded interface
for retrieving recent conversation history. No new tables required.

The configurable `limit` defaults to `MEMORY_MAX_RECENT_MESSAGES` from settings,
ensuring we never send unbounded history to the LLM.
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.services.chat_message_service import ChatMessageService

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Bounded short-term memory — wraps ChatMessageService.

    Interface:
        get_recent_messages(conversation_id, limit) → list[dict]

    Each returned dict has keys: {"role": str, "content": str}.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._service = ChatMessageService(session)

    async def get_recent_messages(
        self,
        conversation_id: uuid.UUID,
        *,
        limit: int | None = None,
        exclude_message_id: uuid.UUID | None = None,
    ) -> list[dict[str, str]]:
        """Return the most recent `limit` messages in a conversation as role/content dicts.

        Args:
            conversation_id: The chat session UUID.
            limit: Max messages to return. Defaults to MEMORY_MAX_RECENT_MESSAGES.
            exclude_message_id: If set, the message with this ID is excluded
                (used to skip the just-created user message that hasn't been
                answered yet).

        Returns:
            List of {role, content} dicts, oldest first.
        """
        settings = get_settings()
        effective_limit = limit if limit is not None else settings.MEMORY_MAX_RECENT_MESSAGES

        # Fetch a slightly larger window to account for potential exclusion
        fetch_limit = effective_limit + (1 if exclude_message_id else 0)

        try:
            messages = await self._service.list_by_session(
                conversation_id, limit=fetch_limit + 2
            )
        except Exception as exc:
            logger.warning(
                "[CONV_MEMORY] Failed to fetch messages session_id=%s error=%s",
                conversation_id,
                exc,
            )
            return []

        # Filter out excluded message and apply window limit
        filtered = [
            m for m in messages
            if exclude_message_id is None or m.id != exclude_message_id
        ]
        # Take the last `effective_limit` messages (most recent)
        recent = filtered[-effective_limit:]

        result = []
        for msg in recent:
            role_str = getattr(msg.role, "value", str(msg.role))
            result.append({"role": role_str, "content": msg.content or ""})

        logger.info(
            "[CONV_MEMORY] session_id=%s returned=%d limit=%d",
            conversation_id,
            len(result),
            effective_limit,
        )
        return result

    async def update_session_summary_if_needed(
        self,
        session_id: uuid.UUID,
        *,
        force: bool = False,
    ) -> str | None:
        """Check if session message count reaches SUMMARY_TRIGGER_MESSAGE_COUNT and generate compressed session summary."""
        settings = get_settings()
        trigger_count = getattr(settings, "SUMMARY_TRIGGER_MESSAGE_COUNT", 6)

        try:
            messages = await self._service.list_by_session(session_id, limit=50)
            if not messages or (len(messages) < trigger_count and not force):
                return None

            from app.models.chat_session import ChatSession
            from sqlalchemy import select
            res = await self._service.repository.session.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            sess_obj = res.scalar_one_or_none()
            if not sess_obj:
                return None

            # Generate summary of important topics, user intent, technical context, decisions, documents discussed
            msg_texts = []
            for m in messages[-20:]:
                role_label = "User" if getattr(m.role, "value", str(m.role)).lower() == "user" else "Assistant"
                text_content = (m.content or "").strip()
                if text_content and not text_content.lower().startswith(("hi", "hello", "hey", "good morning")):
                    msg_texts.append(f"{role_label}: {text_content[:200]}")

            if not msg_texts:
                return None

            user_intents = []
            tech_cues = []
            docs_discussed = []

            for text in msg_texts:
                low = text.lower()
                if "user:" in low:
                    clean_intent = text.replace("User:", "").strip()[:80]
                    if clean_intent:
                        user_intents.append(clean_intent)
                if any(w in low for w in ("ollama", "python", "fastapi", "postgres", "supabase", "rag", "embedding", "qwen", "react")):
                    tech_cues.append(text[:80])
                if any(w in low for w in ("document", "framework", "hr", "policy", "prd", "pdf", "docx")):
                    docs_discussed.append(text[:80])

            summary_parts = []
            if user_intents:
                summary_parts.append(f"User Intent & Topics: {'; '.join(user_intents[-3:])}")
            if tech_cues:
                summary_parts.append(f"Technical Context: {'; '.join(tech_cues[-2:])}")
            if docs_discussed:
                summary_parts.append(f"Documents Discussed: {'; '.join(docs_discussed[-2:])}")

            summary_text = " | ".join(summary_parts)[:getattr(settings, "MAX_SUMMARY_LENGTH", 1500)]

            if summary_text:
                from app.services.chat_session_service import ChatSessionService
                session_service = ChatSessionService(self._service.repository.session)
                await session_service.update_summary(session_id, summary_text)
                logger.info("[SESSION SUMMARY UPDATED] session_id=%s length=%d summary=%r", session_id, len(summary_text), summary_text[:100])
                return summary_text

            return None
        except Exception as exc:
            logger.warning("[SESSION SUMMARY ERROR] session_id=%s error=%s", session_id, exc)
            return None
