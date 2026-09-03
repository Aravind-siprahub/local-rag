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
    "1. Answer ONLY what the user specifically asked for. Focus strictly and exclusively on the exact topic of the question.\n"
    "2. Do NOT include unrequested adjacent topics (e.g., do NOT explain working hours, shift timings, IT security, or codes of conduct when answering a leave question).\n"
    "3. Do NOT add meta-notes, disclaimers, or sections like 'Important Note', 'Note on Provided Context', or 'The document does not mention...'.\n"
    "4. State verified facts from the context directly, clearly, and concisely in clean bullet points.\n"
    "5. Do NOT invent or infer unstated facts, shift times, figures, or policies not present in the document context.\n"
    "6. Do NOT list documents, section numbers, or page numbers.\n"
    "7. Do NOT output self-talk, reasoning, or phrases like 'Let\\'s write', 'We are to be', 'Note that', or 'The key is'.\n"
    "8. If the context contains NO relevant facts whatsoever for the question, say: \"I could not find relevant information in the uploaded documents to answer your question.\"\n"
    "9. Never output internal pipeline explanations (such as how OCR, text detection, PaddleOCR, parsing engines, or RAG works) unless the user explicitly asked about the technical implementation of OCR or parsing.\n"
    "10. If the document context contains technical architecture or OCR specifications from project documents that do not directly answer the user's question, IGNORE them.\n"
    "11. If the retrieved context contains no relevant facts to answer the specific question, do NOT summarize unrelated sections; respond: \"I could not find relevant information in the uploaded documents to answer your question.\"\n"
    "12. PROJECT ISOLATION: When answering for a specific project, application, or system named in the question (such as AIRIS, SipraOne, SipraHub, or Talk to My Data), verify that the retrieved excerpts explicitly describe THAT project. Never borrow, mix in, or attribute technologies, components, or policies from other documents or projects.\n"
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

