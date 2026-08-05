"""Prompt templates for RAG-style question answering."""

CONTEXT_HEADER = "## Document Excerpts"
QUESTION_HEADER = "## Question"
NO_CONTEXT_MESSAGE = "No relevant document excerpts were retrieved."

CHUNK_TEMPLATE = "[Chunk {index}]\n{chunk_text}"

USER_PROMPT_WITH_CONTEXT = (
    "Use the following document excerpts to answer the question. "
    "If the excerpts do not contain enough information, say so.\n\n"
    "{context_header}\n\n"
    "{context}\n\n"
    "{question_header}\n"
    "{question}"
)

USER_PROMPT_WITHOUT_CONTEXT = (
    "No document excerpts were available for this question.\n\n"
    "{question_header}\n"
    "{question}"
)


def format_chunk(index: int, chunk_text: str) -> str:
    """Format one retrieved chunk with a visible index label."""
    return CHUNK_TEMPLATE.format(index=index, chunk_text=chunk_text.strip())


def format_user_prompt(context: str, question: str) -> str:
    """Compose the user message from context blocks and the question."""
    question = question.strip()
    if context.strip():
        return USER_PROMPT_WITH_CONTEXT.format(
            context_header=CONTEXT_HEADER,
            context=context.strip(),
            question_header=QUESTION_HEADER,
            question=question,
        )
    return USER_PROMPT_WITHOUT_CONTEXT.format(question_header=QUESTION_HEADER, question=question)
