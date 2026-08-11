"""Diagnostic: list docs + probe retrieval for failing queries (no route changes)."""
from __future__ import annotations

import asyncio
import sys
import uuid

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.rag.intent_router import classify
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters

USER_ID = uuid.UUID("09e6e22a-36cf-421f-a0f2-8c7950f09a39")

QUERIES = [
    "what tech stack were using for talk to my data",
    "tell frontend and backend what using for talk to my data",
    "AIRIS what tech stack were using tell",
    "AIRIS tech stack",
    "frontend backend talk to my data",
]


async def main() -> None:
    async with AsyncSessionLocal() as session:
        docs = list(
            (
                await session.execute(
                    select(Document)
                    .where(Document.deleted_at.is_(None))
                    .where(Document.user_id == USER_ID)
                )
            ).scalars().all()
        )
        print(f"USER_DOCS={len(docs)}")
        for d in docs:
            status = d.status.value if hasattr(d.status, "value") else str(d.status)
            print(f"  - {d.title} | status={status}")

        if not docs:
            all_docs = list(
                (await session.execute(select(Document).where(Document.deleted_at.is_(None)))).scalars().all()
            )
            print(f"ALL_DOCS={len(all_docs)}")
            for d in all_docs:
                print(f"  - {d.title} | user={d.user_id}")

        retriever = Retriever(session)
        filters = SearchFilters(user_id=USER_ID)
        for q in QUERIES:
            route = classify(q).value
            print(f"\nQ={q!r}")
            print(f"  route={route}")
            try:
                chunks = await retriever.retrieve(q, filters=filters, top_k=5)
            except Exception as exc:
                print(f"  retrieval_error={type(exc).__name__}: {exc}")
                continue
            print(f"  retrieved={len(chunks)}")
            if chunks:
                print(f"  top_similarity={chunks[0].similarity_score:.4f}")
                for i, c in enumerate(chunks[:3]):
                    preview = (c.chunk_text or "")[:90].replace("\n", " ")
                    print(f"  [{i}] sim={c.similarity_score:.4f} doc={c.document_id} preview={preview!r}")


if __name__ == "__main__":
    asyncio.run(main())
