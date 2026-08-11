import asyncio
import sys
import json
import httpx
from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.embedding import Embedding
from app.models.user import User
from app.models.chat_session import ChatSession
from app.repositories.user_repository import UserRepository
from app.repositories.chat_session_repository import ChatSessionRepository
from app.services.chat_session_service import ChatSessionService
from app.services.session_resolution import get_or_create_swagger_demo_session
from app.rag.service import RAGService
from app.retrieval.search import SearchFilters, search_similar
from app.embeddings.client import OllamaEmbeddingClient
from app.prompting.builder import PromptBuilder
from app.llm.ollama_client import OllamaLLMClient
from app.core.config import get_settings

async def main():
    async with AsyncSessionLocal() as session:
        print("=== STEP 0: DATABASE STATE & FILTERS INVENTORY ===")
        users = (await session.execute(select(User))).scalars().all()
        print("Users:", [(str(u.id), u.email) for u in users])

        docs = (await session.execute(select(Document))).scalars().all()
        print("\nDocuments in DB:")
        for d in docs:
            print(f"  Doc ID: {d.id} | Title: {d.title!r} | User ID: {d.user_id} | Status: {d.status} | Deleted: {d.deleted_at}")

        versions = (await session.execute(select(DocumentVersion))).scalars().all()
        print("\nDocument Versions in DB:")
        for v in versions:
            print(f"  Version ID: {v.id} | Doc ID: {v.document_id} | Version #: {v.version_number}")

        chunks = (await session.execute(select(DocumentChunk))).scalars().all()
        print(f"\nDocument Chunks in DB count: {len(chunks)}")
        for c in chunks:
            print(f"  Chunk ID: {c.id} | Version ID: {c.document_version_id} | Index: {c.chunk_index} | Content preview: {c.content[:60]!r}")

        embs = (await session.execute(select(Embedding))).scalars().all()
        print(f"\nEmbeddings in DB count: {len(embs)}")
        for e in embs:
            print(f"  Embedding ID: {e.id} | Chunk ID: {e.chunk_id} | Model: {e.model_name}")

        print("\n=== STEP 1: RESOLVING DEMO CHAT SESSION ===")
        session_id = await get_or_create_swagger_demo_session(
            users=UserRepository(session),
            sessions=ChatSessionRepository(session),
            session_service=ChatSessionService(session),
        )
        chat_sess = (await session.execute(select(ChatSession).where(ChatSession.id == session_id))).scalar_one()
        print(f"Session ID: {session_id} | Owner User ID: {chat_sess.user_id}")

        question = "What is inside PRD_Talk_to_My_Data.docx?"
        print(f"\nQuestion: {question!r}")

        print("\n=== STEP 2: VECTOR EMBEDDING & RETRIEVAL TRACE ===")
        emb_client = OllamaEmbeddingClient()
        q_emb = await emb_client.embed(question)
        print(f"Query embedding generated. Dimensions: {len(q_emb)}")

        settings = get_settings()
        print(f"Embedding model setting: {settings.EMBEDDING_MODEL}")
        print(f"Top K setting: {settings.TOP_K}")
        print(f"Similarity threshold setting: {settings.SIMILARITY_THRESHOLD}")

        # Primary retrieval with user_id filter
        filters_primary = SearchFilters(user_id=chat_sess.user_id)
        hits_primary = await search_similar(
            session,
            q_emb,
            model_name=settings.EMBEDDING_MODEL,
            top_k=settings.TOP_K,
            filters=filters_primary,
        )
        print(f"\nPrimary Search Hits (filters: user_id={chat_sess.user_id}): count = {len(hits_primary)}")

        # Fallback search without user_id filter
        filters_fallback = SearchFilters(user_id=None)
        hits_fallback = await search_similar(
            session,
            q_emb,
            model_name=settings.EMBEDDING_MODEL,
            top_k=settings.TOP_K,
            filters=filters_fallback,
        )
        print(f"\nFallback Search Hits (filters: user_id=None): count = {len(hits_fallback)}")
        for h in hits_fallback:
            sim = 1.0 - h.distance
            print(f"  Hit chunk_id={h.chunk_id} doc={h.document_title!r} dist={h.distance:.4f} sim={sim:.4f}")
            print(f"    Preview: {h.chunk_text[:100]!r}")

        # RAGService full pipeline execution
        print("\n=== STEP 3: FULL RAG PIPELINE EXECUTION (RAGService.ask) ===")
        rag = RAGService(session)
        retrieved_chunks = await rag._retrieve_safely(
            question,
            filters=filters_primary,
            top_k=settings.TOP_K,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
        )

        if not retrieved_chunks and filters_primary.user_id is not None:
            fallback_chunks = await rag._retrieve_safely(
                question,
                filters=SearchFilters(),
                top_k=settings.TOP_K,
                similarity_threshold=settings.SIMILARITY_THRESHOLD,
            )
            if fallback_chunks:
                retrieved_chunks = fallback_chunks

        print(f"\n1. Number of retrieved_chunks: {len(retrieved_chunks)}")
        print("2. Details of retrieved_chunks:")
        for idx, chunk in enumerate(retrieved_chunks, 1):
            doc_title = getattr(chunk, "document_title", "Unknown")
            print(f"   Chunk #{idx}:")
            print(f"     Document title: {doc_title}")
            print(f"     Chunk ID: {chunk.chunk_id}")
            print(f"     Similarity Score: {chunk.similarity_score:.4f}")
            print(f"     First 200 chars:\n{chunk.chunk_text[:200]}")

        print("\n=== STEP 4: PROMPT BUILDING TRACE ===")
        pb = PromptBuilder()
        prompt = pb.build(question, retrieved_chunks)
        print("\n3. Complete USER PROMPT sent to Ollama:")
        print("----------------------------------------")
        print(prompt.user_prompt)
        print("----------------------------------------")

        print("\n4. Complete SYSTEM PROMPT sent to Ollama:")
        print("----------------------------------------")
        print(prompt.system_prompt)
        print("----------------------------------------")

        print("\n=== STEP 5: OLLAMA PAYLOAD & HTTP INTERCEPTION ===")
        llm = OllamaLLMClient()
        options = llm._build_options()
        payload = {
            "model": llm.model,
            "messages": [
                {"role": "system", "content": prompt.system_prompt.strip()},
                {"role": "user", "content": prompt.user_prompt.strip()},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": options,
        }
        print("5. Exact request payload sent to Ollama:")
        print(json.dumps(payload, indent=2))

        print("\n=== STEP 6: RAW OLLAMA RESPONSE ===")
        url = f"{llm.base_url}/api/chat"
        async with httpx.AsyncClient(timeout=httpx.Timeout(llm.timeout)) as http_client:
            res = await http_client.post(url, json=payload)
            print(f"Ollama HTTP status: {res.status_code}")
            raw_response_data = res.json()
            print("6. Raw response returned by Ollama:")
            print(json.dumps(raw_response_data, indent=2))

            from app.llm.ollama_client import _parse_chat_response
            parsed_llm_response = _parse_chat_response(raw_response_data, llm.model)
            print(f"\nParsed Answer returned to User:\n{parsed_llm_response.answer}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
