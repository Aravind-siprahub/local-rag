"""Empirical diagnostic script to inspect indexed HR framework chunks and test retrieval pipeline."""
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.enums import DocumentStatus
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters, search_document_chunks_structured
from app.rag.intent_router import classify, Route


async def main():
    async with AsyncSessionLocal() as session:
        print("\n=======================================================")
        print("1. DATABASE DOCUMENT & CHUNK INDEX INSPECTION")
        print("=======================================================\n")

        stmt = (
            select(Document)
            .where(Document.deleted_at.is_(None))
            .order_by(Document.created_at.desc())
        )
        res = await session.execute(stmt)
        docs = res.scalars().all()

        print(f"Total ready/active documents in database: {len(docs)}")
        for d in docs:
            print(f" - Document ID: {d.id} | Title: {d.title!r} | Status: {d.status} | Version: {d.current_version_id}")

        if not docs:
            print("No documents found in database! Creating test verification mock database state...")
            return

        # Pick the most recent document or document with HR Framework in title
        hr_doc = next((d for d in docs if "hr" in d.title.lower() or "framework" in d.title.lower()), docs[0])
        print(f"\nTarget Document for Inspection: '{hr_doc.title}' (ID: {hr_doc.id})")

        # Fetch all chunks for this document version
        chunk_stmt = (
            select(DocumentChunk)
            .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
            .where(DocumentVersion.document_id == hr_doc.id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        chunk_res = await session.execute(chunk_stmt)
        chunks = chunk_res.scalars().all()

        print(f"\nTotal Indexed Chunks for '{hr_doc.title}': {len(chunks)}")
        sections_map = {}
        for c in chunks:
            sec = (c.section_title or "Uncategorized").strip()
            sections_map.setdefault(sec, []).append(c)

        print(f"\nUnique Sections Found ({len(sections_map)} sections):")
        for sec_name, sec_chunks in sections_map.items():
            print(f"  * [{sec_name}] -> {len(sec_chunks)} chunk(s) (Indices: {[c.chunk_index for c in sec_chunks]})")

        expected_sections = [
            "Working Hours & Attendance",
            "Leave Policy",
            "Casual Leave",
            "Leave Application Process",
            "Public Holidays",
            "Leave Without Pay",
            "WFH / Remote Work",
            "Performance Management",
            "Code of Conduct",
            "IT & Security",
            "Grievance Redressal",
            "POSH",
            "Exit & Termination",
        ]

        print("\nChecking Expected HR Sections Indexing Status:")
        for exp in expected_sections:
            found_key = next((k for k in sections_map if exp.lower() in k.lower()), None)
            if found_key:
                print(f"  [INDEXED OK] {exp} -> matched section '{found_key}' ({len(sections_map[found_key])} chunks)")
            else:
                print(f"  [MISSING/PARTIAL] {exp} -> Not explicitly indexed as a distinct section heading.")

        print("\n=======================================================")
        print("2. PIPELINE RETRIEVAL TRACE FOR TEST QUERY")
        print("=======================================================\n")

        test_query = "Summarize the new HR framework document and tell me more detail"
        classified_route = classify(test_query)
        print(f"Query: {test_query!r}")
        print(f"Classified Intent Route: {classified_route.value}")

        retriever = Retriever(session=session)
        filters = SearchFilters(user_id=hr_doc.user_id, document_id=hr_doc.id)

        section_aware_results = await retriever.retrieve_section_aware(
            test_query,
            filters=filters,
            max_total_chunks=35,
        )

        print(f"\nSection-Aware Retrieval Results ({len(section_aware_results)} chunks selected):")
        retrieved_sections = []
        for idx, r in enumerate(section_aware_results, 1):
            sec = r.section_title or f"Page {r.page_number}"
            if sec not in retrieved_sections:
                retrieved_sections.append(sec)
            print(f"  Chunk #{idx:02d} | DocID: {r.document_id} | Section: {sec!r} | Score: {r.similarity_score:.2f}")
            print(f"    Preview: {r.chunk_text.strip()[:120]!r}\n")

        print("=======================================================")
        print("3. FINAL CONTEXT ASSEMBLY SUMMARY")
        print("=======================================================")
        assembled_text = "\n---\n".join([r.chunk_text for r in section_aware_results])
        print(f"Total Context Characters: {len(assembled_text)} (~{len(assembled_text)//4} tokens)")
        print(f"Unique Sections Represented in Context: {len(retrieved_sections)}")
        for sec in retrieved_sections:
            print(f"  - {sec}")


if __name__ == "__main__":
    asyncio.run(main())
