import sys
sys.path.append("C:/Users/ARAVIND/Desktop/local-rag/backend")
import asyncio
import re

def _validate_web_answer(
    raw_answer: str,
    clean_answer: str,
    concise_text: str,
    original_query: str,
) -> str:
    import json as _json

    raw = (raw_answer or "").strip()

    if raw.startswith("{") and "answer" in raw:
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                extracted = (parsed.get("answer") or "").strip()
                if extracted:
                    return extracted
                return concise_text
        except (_json.JSONDecodeError, ValueError):
            return concise_text

    if not clean_answer:
        return concise_text

    import re as _re
    query_tokens = set(_re.findall(r"\b[A-Za-z]{4,}\b", original_query.lower()))
    web_tokens = set(_re.findall(r"\b[A-Za-z]{4,}\b", concise_text.lower()))
    stopwords = {"what", "when", "where", "which", "that", "with", "from", "this", "they", "have", "here", "found"}
    topic_tokens = (query_tokens | web_tokens) - stopwords

    if topic_tokens:
        ans_lower = clean_answer.lower()
        overlap = sum(1 for t in topic_tokens if t in ans_lower)
        if overlap / len(topic_tokens) < 0.15:
            return concise_text

    return clean_answer

raw_answer = "Python is a programming language."
clean_answer = "Python is a programming language."
concise_text = "Here is what I found:\n1. Good Friday 2026: Good Friday in 2026 falls on Friday, 3 April 2026. (https://example.com/good-friday-2026)"
original_query = "When is Good Friday in 2026?"

print("Tokens calculation:")
query_tokens = set(re.findall(r"\b[A-Za-z]{4,}\b", original_query.lower()))
web_tokens = set(re.findall(r"\b[A-Za-z]{4,}\b", concise_text.lower()))
stopwords = {"what", "when", "where", "which", "that", "with", "from", "this", "they", "have", "here", "found"}
topic_tokens = (query_tokens | web_tokens) - stopwords
print(f"topic_tokens: {topic_tokens}")

res = _validate_web_answer(raw_answer, clean_answer, concise_text, original_query)
print(f"Result: {res}")
