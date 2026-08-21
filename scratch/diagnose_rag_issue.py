import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath("backend"))

from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.embedding import Embedding
from app.repositories.user_repository import UserRepository
from app.rag.intent_router import classify
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters, search_similar, search_fulltext
from app.embeddings.client import OllamaEmbeddingClient
from app.core.config import get_settings

async def main():
    query = "what fronted and backend are using talk to my data"
    print("=" * 80)
    print("RAG PIPELINE DIAGNOSTIC REPORT")
    print("=" * 80)
    
    async with AsyncSessionLocal() as session:
        settings = get_settings()
        print(f"Embedding Model Config: {settings.EMBEDDING_MODEL}")
        print(f"Top K: {settings.TOP_K}")
        print(f"Similarity Threshold: {settings.SIMILARITY_THRESHOLD}")
        
        # 1. Users
        user_repo = UserRepository(session)
        users = await user_repo.list_active()
        print(f"\n[1. USERS] Active user count: {len(users)}")
        for u in users:
            print(f"  User ID: {u.id} | Email: {u.email}")

        # 2. Documents
        docs_res = await session.execute(select(Document))
        docs = docs_res.scalars().all()
        print(f"\n[2. DOCUMENTS] Total document count: {len(docs)}")
        for d in docs:
            print(f"  Doc ID: {d.id} | Title: {d.title!r} | Status: {d.status} | Deleted: {d.deleted_at} | User ID: {d.user_id} | CurrVer: {d.current_version_id}")

        # 3. Chunks & Embeddings Count
        chunk_count = (await session.execute(select(func.count()).select_from(DocumentChunk))).scalar_one()
        emb_count = (await session.execute(select(func.count()).select_from(Embedding))).scalar_one()
        print(f"\n[3. STORAGE] Total Chunks: {chunk_count} | Total Embeddings: {emb_count}")

        # Embeddings grouped by model name
        emb_models = await session.execute(
            select(Embedding.model_name, func.count()).group_by(Embedding.model_name)
        )
        for model_name, count in emb_models.all():
            print(f"  Model in DB: {model_name!r} -> Count: {count}")

        # 4. Intent Routing
        route = classify(query)
        print(f"\n[4. ROUTING] Query: {query!r} -> Route: {route.name} ({route.value})")

        # 5. Test Retrieval per user
        for u in users:
            print(f"\n[5. RETRIEVAL TEST for User {u.email} ({u.id})]")
            filters = SearchFilters(user_id=u.id)
            retriever = Retriever(session)
            results = await retriever.retrieve(query, filters=filters)
            print(f"  Retriever returned {len(results)} chunks")
            for idx, r in enumerate(results, 1):
                print(f"    {idx}. doc={r.document_title!r} sim={r.similarity_score:.4f} text={r.chunk_text[:100]!r}")

        # Test Retrieval without user_id filter
        print(f"\n[6. RETRIEVAL TEST without user_id filter]")
        retriever_global = Retriever(session)
        results_global = await retriever_global.retrieve(query, filters=SearchFilters())
        print(f"  Global Retriever returned {len(results_global)} chunks")
        for idx, r in enumerate(results_global, 1):
            print(f"    {idx}. doc={r.document_title!r} sim={r.similarity_score:.4f} text={r.chunk_text[:100]!r}")

if __name__ == "__main__":
    asyncio.run(main())
