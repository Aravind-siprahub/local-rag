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
    "1. Give a direct, helpful, factual, and complete response summarizing all relevant details present in the context related to the user's question.\n"
    "2. If the question asks to explain or define a general concept, acronym, or industry term (such as 'POC' / 'Proof of Concept'), first define the general concept clearly, and then explain how that concept is specifically used or applied in the document context above.\n"
    "3. State all verified facts, tracking rules, policies, and metrics present in the document context accurately. If the question asks for specific details (such as fixed shift start/end times, lunch breaks, or exact hours) that are NOT explicitly mentioned in the context, state what the document DOES record while clarifying that exact fixed times or figures are not explicitly specified.\n"
    "4. Inspect the uploaded document context for matching keywords or concepts from the question. When matching keywords or sections are found, extract and state the full answer directly based on those matching document details.\n"
    "5. Do NOT invent or infer unstated facts, shift times, figures, or policies not present in the document context.\n"
    "6. Do NOT list documents, section numbers, or page numbers.\n"
    "7. Do NOT output self-talk, reasoning, or phrases like 'Let's write', 'We are to be', 'Note that', or 'The key is'.\n"
    "8. If the context contains NO relevant facts whatsoever for the question, say: \"The requested information is not found in the documents.\"\n"
)

USER_PROMPT_WITHOUT_CONTEXT = (
    "Retrieved Document Context:\n\n"
    "No document excerpts were available.\n\n"
    "Question:\n\n{question}\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "No document context was retrieved for this query. If the question asks for information from uploaded documents or project specifications, or if no relevant context exists, respond: \"Information not found in document excerpts.\""
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
    long_term_memory_context: str | None = None,
    max_history_chars: int = 1000,
) -> str:
    """Compose the user message from context blocks, working memory summary, and the question.

    Order of sections (all optional):
        1. Working memory summary (short-term rolling summary)
        2. Long-term memory context (DATA block, prompt-injection safe)
        3. RAG document context
        4. Question + grounding rules
    """
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

    # Build the base prompt from RAG context
    if context.strip():
        base = USER_PROMPT_WITH_CONTEXT.format(
            question=question,
            context=context.strip(),
            context_header="Retrieved Document Context",
            question_header="Question:",
        )
    else:
        base = USER_PROMPT_WITHOUT_CONTEXT.format(question=question)

    # Assemble sections in order: history → long-term memory → base
    parts: list[str] = []
    if history_text:
        parts.append(history_text)

    if long_term_memory_context and long_term_memory_context.strip():
        parts.append(long_term_memory_context.strip())
        parts.append("\n")  # spacer before RAG context

    parts.append(base)
    return "".join(parts)

