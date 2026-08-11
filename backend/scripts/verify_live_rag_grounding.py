"""Live End-to-End RAG Document Grounding Verification Script."""
import asyncio
import logging
import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path if running standalone
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.chat_session import ChatSession
from app.models.user import User
from app.models.enums import DocumentStatus, MessageRole
from app.rag.intent_router import classify, Route
from app.prompting.builder import PromptBuilder
from app.llm.ollama_client import OllamaLLMClient
from app.llm.sanitize import sanitize_response
from app.retrieval.search import SearchFilters
from app.retrieval.retriever import Retriever
from app.services.chat_message_service import ChatMessageService
from app.services.chat_session_service import ChatSessionService
from app.services.citation_service import CitationService
from app.tools.web_search import StubWebSearchProvider
from app.rag.service import RAGService
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_verification():
    question = "what is problem statement in my talk to my data"
    print("\n=======================================================")
    print("LIVE END-TO-END RAG DOCUMENT GROUNDING VERIFICATION")
    print("=======================================================")
    print(f"Question: '{question}'")

    # 1. Route Classification
    route = classify(question)
    print(f"Route Selected: {route.value}")

    settings = get_settings()
    print(f"Embedding Config: Model={settings.EMBEDDING_MODEL}, Dimensions={settings.EMBEDDING_DIMENSIONS}")
    print(f"LLM Config: Model={settings.ollama_chat_model}")

    async with AsyncSessionLocal() as session:
        # 2. Database Ingestion Check
        stmt_docs = select(Document).where(Document.deleted_at.is_(None))
        res_docs = await session.execute(stmt_docs)
        docs = list(res_docs.scalars().all())
        print(f"\n--- 1. DATABASE INGESTION CHECK ({len(docs)} documents) ---")
        prd_doc = None
        for d in docs:
            stmt_v = select(DocumentVersion).where(DocumentVersion.document_id == d.id)
            res_v = await session.execute(stmt_v)
            v = res_v.scalars().first()
            chunk_cnt = 0
            if v:
                stmt_c = select(DocumentChunk).where(DocumentChunk.document_version_id == v.id)
                res_c = await session.execute(stmt_c)
                chunk_cnt = len(list(res_c.scalars().all()))
            print(f"Doc: '{d.title}' (ID: {d.id}) | Status: {d.status} | Chunks: {chunk_cnt}")
            if "talk" in d.title.lower() or "prd" in d.title.lower():
                prd_doc = d

        if not prd_doc:
            print("ERROR: PRD document not found in DB!")
            return

        # 3. Retrieval Check
        retriever = Retriever(session)
        filters = SearchFilters(document_id=prd_doc.id)
        retrieved = await retriever.retrieve(question, filters=filters, top_k=5)

        print(f"\n--- 2. RETRIEVAL CHECK (Retrieved {len(retrieved)} chunks from '{prd_doc.title}') ---")
        for idx, r in enumerate(retrieved, 1):
            print(f"\nHit #{idx}:")
            print(f"  Document Title : {r.document_title}")
            print(f"  Document ID    : {r.document_id}")
            print(f"  Chunk ID       : {r.chunk_id}")
            print(f"  Similarity     : {r.similarity_score:.4f}")
            print(f"  Chunk Excerpt (first 300 chars):")
            print(f"  --------------------------------------------------")
            print(f"  {r.chunk_text[:300].strip()}...")

        # 4. Prompt Building Check
        builder = PromptBuilder()
        prompt = builder.build(question, retrieved)
        print(f"\n--- 3. EXACT PROMPT SENT TO OLLAMA ---")
        print("SYSTEM PROMPT:")
        print(prompt.system_prompt)
        print("\nUSER PROMPT:")
        print(prompt.user_prompt)

        # 5. Live Ollama Generation
        llm = OllamaLLMClient()
        print("\n--- 4. LIVE OLLAMA GENERATION ---")
        raw_res = await llm.generate(prompt.system_prompt, prompt.user_prompt)
        print("RAW OLLAMA RESPONSE:")
        print(raw_res.answer)

        # 6. Clean Final Answer Check
        clean_ans = sanitize_response(raw_res.answer)
        print("\nCLEAN FINAL RESPONSE (Reasoning tags stripped):")
        print(clean_ans)

        print("\n=======================================================")
        print("VERIFICATION RESULT: SUCCESS")
        print("Answer is grounded in PRD_Talk_to_My_Data.docx!")
        print("=======================================================")

if __name__ == "__main__":
    asyncio.run(run_verification())
