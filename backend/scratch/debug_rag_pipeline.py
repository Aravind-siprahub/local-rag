"""
Debug script: trace the FULL RAG pipeline for the target query.
Identifies exactly where correct information is lost or changed.

Run from backend/ directory:
    python scratch/debug_rag_pipeline.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUERY = "What frontend and backend are used in Talk to My Data?"

# ─── 1. test the intent router ─────────────────────────────────────────────
from app.rag.intent_router import classify
from app.rag.query_normalizer import normalize_query

def test_routing():
    print("\n" + "="*70)
    print("STAGE 1: INTENT ROUTER")
    print("="*70)
    orig, norm, ret = normalize_query(QUERY)
    print(f"  Original query : {orig!r}")
    print(f"  Normalized     : {norm!r}")
    print(f"  Retrieval query: {ret!r}")

    # Without document titles  
    route_no_docs = classify(QUERY, document_titles=None)
    print(f"\n  Route (no docs): {route_no_docs.value}")

    # With document titles matching the user's corpus
    route_with_docs = classify(QUERY, document_titles=["PRD_Talk_to_My_Data.docx"])
    print(f"  Route (with doc titles): {route_with_docs.value}")

    if route_with_docs.value != "DOCUMENT_QA":
        print("  XBUG: Intent router is NOT routing to DOCUMENT_QA!")
        print("     This means the query will bypass RAG retrieval entirely.")
    else:
        print("  OK: Intent router correctly selects DOCUMENT_QA")

    return orig, norm, ret


# ─── 2. test _parse_user_prompt (the bug suspect) ──────────────────────────
def test_parse_user_prompt():
    print("\n" + "="*70)
    print("STAGE 2: _parse_user_prompt (ollama_client.py)")
    print("="*70)

    from app.llm.ollama_client import _parse_user_prompt
    from app.prompting.templates import USER_PROMPT_WITH_CONTEXT, format_user_prompt, format_chunk

    # Simulate the exact prompt the pipeline builds
    chunk1_text = "Frontend -- the chat interface; also renders citations, source snippets."
    chunk2_text = "Backend -- FastAPI handles API requests, orchestrates RAG, and manages PostgreSQL connections."

    chunk1 = format_chunk(1, chunk1_text, title="PRD_Talk_to_My_Data.docx", section="System Architecture Overview", page=5)
    chunk2 = format_chunk(2, chunk2_text, title="PRD_Talk_to_My_Data.docx", section="System Architecture Overview", page=5)
    context = chunk1 + "\n\n" + chunk2

    user_prompt = format_user_prompt(context, QUERY)
    print(f"\n  user_prompt (first 400 chars):\n{user_prompt[:400]!r}\n")

    history_msgs, ctx_text, actual_query = _parse_user_prompt(user_prompt)

    print(f"  Parsed history messages: {len(history_msgs)}")
    print(f"  Parsed context (first 200 chars): {(ctx_text or 'NONE')[:200]!r}")
    print(f"  Parsed actual_query: {actual_query!r}")

    if not ctx_text or "Frontend" not in ctx_text:
        print("\n  X BUG: _parse_user_prompt is NOT extracting context text!")
        print("     Ollama will receive ONLY the question with NO document passages.")
    else:
        print("\n  OK: Context text correctly extracted, contains document chunks")

    return ctx_text


# ─── 3. test the Ollama payload builder  ───────────────────────────────────
def test_ollama_payload():
    print("\n" + "="*70)
    print("STAGE 3: Ollama _build_payload")
    print("="*70)

    from app.llm.ollama_client import OllamaLLMClient
    from app.prompting.templates import format_user_prompt, format_chunk
    from app.core.config import get_settings

    settings = get_settings()

    chunk1_text = "Frontend -- the chat interface; also renders citations, source snippets."
    chunk2_text = "Backend -- FastAPI handles API requests, orchestrates RAG, and manages PostgreSQL connections."

    chunk1 = format_chunk(1, chunk1_text, title="PRD_Talk_to_My_Data.docx", section="System Architecture Overview", page=5)
    chunk2 = format_chunk(2, chunk2_text, title="PRD_Talk_to_My_Data.docx", section="System Architecture Overview", page=5)
    context = chunk1 + "\n\n" + chunk2

    system_prompt = settings.SYSTEM_PROMPT
    user_prompt = format_user_prompt(context, QUERY)

    client = OllamaLLMClient()
    payload = client._build_payload(system_prompt, user_prompt, stream=False)

    messages = payload.get("messages", [])
    print(f"\n  Model: {payload.get('model')}")
    print(f"  think: {payload.get('think', 'NOT SET')}")
    print(f"  temperature: {payload['options'].get('temperature')}")
    print(f"  num_predict: {payload['options'].get('num_predict')}")
    print(f"  num_ctx: {payload['options'].get('num_ctx')}")
    print(f"\n  Messages in payload: {len(messages)}")

    all_content = ""
    for i, msg in enumerate(messages):
        role = msg.get('role')
        content = msg.get('content', '')
        all_content += content
        has_chunk = "Frontend" in content or "Backend" in content
        print(f"\n  [msg{i}] role={role!r} len={len(content)} has_doc_content={has_chunk}")
        print(f"    {content[:250]!r}")

    if "Frontend" not in all_content:
        print("\n  X CRITICAL BUG: Document chunks NOT in Ollama payload! Model sees no context.")
    else:
        print("\n  OK: Document chunks found in Ollama payload")


if __name__ == "__main__":
    print("RAG PIPELINE DEBUG")
    print("Query:", QUERY)

    test_routing()
    ctx = test_parse_user_prompt()
    test_ollama_payload()

    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)
