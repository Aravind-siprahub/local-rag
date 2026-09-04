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
    "CRITICAL GROUNDING RULES:\n"
    "1. The retrieved document context is your ONLY source of truth. Pretrained knowledge is strictly forbidden for document-based questions.\n"
    "2. MULTI-PART QUESTIONS: If the question asks about multiple topics (e.g. topic A and topic B), you MUST address EVERY requested topic:\n"
    "   - For topics present in the context: Answer factually using ONLY the provided excerpts.\n"
    "   - For topics NOT present in the context: Explicitly state: \"The provided document does not specify [Topic].\"\n"
    "   - NEVER omit a requested topic, and NEVER invent policies or numbers to fill in an absent topic.\n"
    "3. EXACT NUMBERS & TERMINOLOGY: Preserve exact wording, numbers, limits, and time periods from the document (e.g. if the document states \"1 (one) Casual Leave per month\", state exactly that; do NOT change it to \"12 casual leaves annually\").\n"
    "4. NO POLICY SUBSTITUTION: Do NOT combine or substitute unrelated sections (e.g. do NOT substitute Code of Conduct for Core Values unless explicitly asked).\n"
    "5. RELEVANCE: Answer ONLY what the user asked. Do NOT include unrequested adjacent topics (e.g. do not explain working hours or IT security when asked about leave).\n"
    "6. FORMAT: Present verified facts directly, clearly, and concisely in clean bullet points without self-talk or reasoning monologue.\n"
    "7. COMPLETELY UNSUPPORTED: If the document context contains no supporting information for the question or is insufficient, respond: \"I couldn't find enough information in the available documents to answer this question.\"\n"
    "8. PROJECT ISOLATION: When answering for a specific project (such as AIRIS, SipraOne, SipraHub, or Talk to My Data), verify that the retrieved excerpts explicitly describe THAT project. Never mix in or attribute technologies or policies from other documents or projects.\n"
)

USER_PROMPT_WITHOUT_CONTEXT = (
    "Retrieved Document Context:\n\n"
    "No document excerpts were available.\n\n"
    "Question:\n\n{question}\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "No document context was retrieved for this query. If the question asks for information from uploaded documents or project specifications, or if no relevant context exists, respond: \"I couldn't find enough information in the available documents to answer this question.\""
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

