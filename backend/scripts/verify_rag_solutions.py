"""Verification script for Leave Policy, Core Values, and regression queries."""
from __future__ import annotations

import asyncio
import io
import sys
from dataclasses import dataclass
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8")

if sys.platform == "win32" and sys.version_info < (3, 14):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[deprecated]

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters


@dataclass(frozen=True)
class QueryTestCase:
    """Strongly typed query test case definition."""
    query: str
    doc_match: str
    expected_terms: list[str]


TEST_QUERIES: list[QueryTestCase] = [
    QueryTestCase(
        query="what are Leave Policy in Siprahub ?",
        doc_match="%HR Framework%",
        expected_terms=["casual leave", "1 (one) casual leave", "carry forward", "leave utilization"],
    ),
    QueryTestCase(
        query="what are core values of Siprahub ?",
        doc_match="%HR Framework%",
        expected_terms=["integrity", "accountability", "collaboration", "excellence", "respect"],
    ),
    QueryTestCase(
        query="what tech stack were using for talk to my data",
        doc_match="%talk%",
        expected_terms=["frontend", "backend", "duckdb"],
    ),
]


async def verify() -> None:
    async with AsyncSessionLocal() as session:
        retriever = Retriever(session)

        all_passed = True
        for item in TEST_QUERIES:
            print("\n" + "=" * 80)
            print(f"TESTING: {item.query}")
            print("=" * 80)

            # Find matching doc
            stmt = (
                select(Document)
                .where(Document.deleted_at.is_(None))
                .where(Document.title.ilike(item.doc_match))
            )
            res = await session.execute(stmt)
            doc = res.scalars().first()

            if not doc:
                print(f"⚠️ Warning: No document found matching {item.doc_match!r}")
                filters = None
            else:
                filters = SearchFilters(document_id=doc.id)

            hits = await retriever.retrieve(item.query, filters=filters, top_k=3)

            print(f"Retrieved {len(hits)} hit(s):")
            combined_text = ""
            for i, h in enumerate(hits, 1):
                combined_text += " " + h.chunk_text.lower()
                print(f"  Hit #{i}: score={h.similarity_score:.4f}, section={h.section_title!r}")
                print(f"    Excerpt: {h.chunk_text[:200].replace(chr(10), ' ')}...")

            # Check expected terms
            missing = [t for t in item.expected_terms if t.lower() not in combined_text]
            if missing:
                print(f"❌ FAILED: Missing expected terms: {missing}")
                all_passed = False
            else:
                print(f"✅ PASSED: All expected terms found: {item.expected_terms}")

        print("\n" + "=" * 80)
        if all_passed:
            print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        else:
            print("⚠️ SOME TESTS FAILED!")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(verify())
