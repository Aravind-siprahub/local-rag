"""Working Memory Summarizer for Local RAG sessions.

Compresses multi-turn conversation history into a compact working memory summary (<150 words)
to reduce prompt token usage and keep context lightweight for small local LLMs.
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


def summarize_session_history(
    chat_history: list[dict[str, str]],
    existing_summary: str | None = None,
) -> str:
    """Compress multi-turn chat history into a concise working memory summary (<150 words)."""
    if not chat_history:
        return existing_summary or ""

    user_intents: list[str] = []
    project_entities: set[str] = set()
    key_topics: set[str] = set()

    for msg in chat_history:
        role = msg.get("role", "").lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        content_lower = content.lower()

        # Extract project entities
        if "talk to my data" in content_lower:
            project_entities.add("Talk to My Data")
        if "sipraone" in content_lower:
            project_entities.add("SipraOne")
        if "siprahub" in content_lower:
            project_entities.add("SipraHub")

        # Extract user intent topics
        if role == "user":
            if "frontend" in content_lower or "backend" in content_lower or "tech stack" in content_lower:
                key_topics.add("Tech Stack & Architecture")
            if "port" in content_lower or "nginx" in content_lower or "pm2" in content_lower:
                key_topics.add("Ports & Deployment")
            if "policy" in content_lower or "leave" in content_lower:
                key_topics.add("Company Policy")

            # Extract short user goal statement
            clean_user = re.sub(r"[\n\r]+", " ", content)[:80].strip()
            if clean_user and len(user_intents) < 3:
                user_intents.append(clean_user)

    summary_parts: list[str] = []

    if project_entities:
        summary_parts.append(f"Active Project: {', '.join(sorted(project_entities))}.")

    if key_topics:
        summary_parts.append(f"Discussed Topics: {', '.join(sorted(key_topics))}.")

    if user_intents:
        summary_parts.append(f"Recent Queries: {'; '.join(user_intents)}.")

    if existing_summary and existing_summary.strip():
        # Preserve core context from existing summary if not already present
        clean_existing = existing_summary.strip()
        if clean_existing not in summary_parts:
            summary_parts.insert(0, clean_existing)

    full_summary = " ".join(summary_parts).strip()
    words = full_summary.split()
    if len(words) > 150:
        full_summary = " ".join(words[:150]) + "..."

    return full_summary
