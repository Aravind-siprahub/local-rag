from typing import Any

CHUNK_TEMPLATE = (
    "---\n"
    "Document: {title}\n"
    "Section: {section}\n"
    "Page: {page}\n\n"
    "{chunk_text}"
)

USER_PROMPT_WITH_CONTEXT = (
    "{context_header}:\n\n"
    "{context}\n\n"
    "{question_header} {question}\n\n"
    "CRITICAL RULES FOR YOUR RESPONSE:\n"
    "1. Give ONLY the direct factual answer immediately (e.g. \"Frontend: React, Backend: FastAPI\").\n"
    "2. Do NOT list documents, section numbers, or page numbers.\n"
    "3. Do NOT output self-talk, reasoning, or phrases like 'Let's write', 'We are to be', 'Note that', or 'The key is'.\n"
    "4. If information is not found in the context, reply ONLY: \"The requested information is not found in the documents.\"\n"
)

USER_PROMPT_WITHOUT_CONTEXT = (
    "Question:\n\n{question}\n\n"
    "---------------------------------\n\n"
    "Retrieved Document Context\n\nNo document excerpts were available."
)


def format_chunk(
    index: int,
    chunk_text: str,
    title: str = "Unknown",
    section: str = "N/A",
    page: int | str = "N/A",
    chunk_id: Any = "N/A",  # kept for API compatibility; no longer rendered in template
) -> str:
    """Format one retrieved passage with document title, section, and page for the LLM prompt."""
    return CHUNK_TEMPLATE.format(
        title=title.strip(),
        section=section.strip() if section else "General",
        page=str(page) if page is not None else "1",
        chunk_text=chunk_text.strip(),
    )


def format_user_prompt(
    context: str,
    question: str,
    chat_history: list[dict[str, str]] | None = None,
    working_memory_summary: str | None = None,
    max_history_chars: int = 1000,
) -> str:
    """Compose the user message from context blocks, working memory summary, and the question."""
    question = question.strip()
    history_text = ""

    if working_memory_summary and working_memory_summary.strip():
        history_text = f"Working Memory Context:\n<working_memory_summary>\n{working_memory_summary.strip()}\n</working_memory_summary>\n\n---------------------------------\n\n"
    elif chat_history:
        recent = chat_history[-6:]
        total_chars = sum(len(m.get("content", "")) for m in recent)

        formatted_history = []
        if total_chars > max_history_chars and len(recent) > 2:
            # Compress older turns (1..N-2) into a 1-line rolling summary block
            older_turns = recent[:-2]
            newer_turns = recent[-2:]
            summarized_topics = []
            for m in older_turns:
                content = m.get("content", "").strip()
                if content:
                    first_line = content.split("\n")[0][:60]
                    summarized_topics.append(first_line)
            if summarized_topics:
                formatted_history.append(f"[Prior Conversation Summary: {'; '.join(summarized_topics)}...]")

            for msg in newer_turns:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted_history.append(f"{role}: {msg.get('content', '').strip()}")
        else:
            for msg in recent:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted_history.append(f"{role}: {msg.get('content', '').strip()}")

        if formatted_history:
            history_text = "Recent Conversation:\n" + "\n".join(formatted_history) + "\n\n---------------------------------\n\n"

    if context.strip():
        base = USER_PROMPT_WITH_CONTEXT.format(
            question=question,
            context=context.strip(),
            context_header="Retrieved Document Context",
            question_header="Question:",
        )
    else:
        base = USER_PROMPT_WITHOUT_CONTEXT.format(question=question)

    if history_text:
        return f"{history_text}{base}"
    return base

