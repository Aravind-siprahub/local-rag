"""Memory Tool for retrieving short-term conversation context and working memory summaries."""
from __future__ import annotations

import logging
import time
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.base import Tool, ToolInput, ToolMetadata, ToolOutput
from app.services.chat_message_service import ChatMessageService
from app.services.chat_session_service import ChatSessionService
from app.models.enums import MessageRole

logger = logging.getLogger(__name__)


class MemoryTool(Tool):
    """Modular tool for memory and context retrieval."""

    def __init__(self, session: AsyncSession | None = None) -> None:
        super().__init__(
            ToolMetadata(
                name="memory_context",
                description="Retrieves working memory summary and bounded short-term conversation history.",
                version="1.0.0",
            )
        )
        self.session = session
        self.message_service = ChatMessageService(session) if session is not None else None
        self.session_service = ChatSessionService(session) if session is not None else None

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        start_mono = time.monotonic()
        params = tool_input.parameters
        session_id = params.get("session_id")
        exclude_message_id = params.get("exclude_message_id")
        history_limit = params.get("history_limit", 4)

        if not session_id or self.session_service is None or self.message_service is None:
            return ToolOutput(
                success=True,
                data={"history": [], "working_memory": None},
                execution_time_ms=0,
            )

        try:
            chat_session = await self.session_service.get(session_id)
            working_memory = getattr(chat_session, "working_memory_summary", None) if chat_session else None

            recent_msgs = await self.message_service.list_by_session(session_id, limit=history_limit + 2)
            prior_msgs = [m for m in recent_msgs if getattr(m, "id", None) != exclude_message_id][-history_limit:]

            history_dicts = []
            for m in prior_msgs:
                role_str = getattr(m.role, "value", str(m.role))
                history_dicts.append({"role": role_str, "content": m.content})

            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.info(
                "[MEMORY TOOL SUCCESS] session_id=%s history_count=%d has_working_mem=%s duration_ms=%d",
                session_id, len(history_dicts), bool(working_memory), duration_ms
            )

            return ToolOutput(
                success=True,
                data={
                    "history": history_dicts,
                    "working_memory": working_memory,
                    "count": len(history_dicts),
                },
                execution_time_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.exception("[MEMORY TOOL FAILED] session_id=%s error=%s", session_id, exc)
            return ToolOutput(
                success=False,
                data={"history": [], "working_memory": None},
                error=str(exc),
                execution_time_ms=duration_ms,
            )
