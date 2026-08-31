"""End-to-End Validation Runner for Local RAG Document Summary and Factual Queries."""
import asyncio
import logging
import uuid
import sys

logging.basicConfig(level=logging.WARNING)

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.rag.service import RAGService
from app.rag.intent_router import classify, Route
from app.services.chat_session_service import ChatSessionService
from app.services.chat_message_service import ChatMessageService
from app.retrieval.search import SearchFilters


async def run_e2e_validation():
    async with AsyncSessionLocal() as session:
        session_svc = ChatSessionService(session)
        msg_svc = ChatMessageService(session)
        rag_svc = RAGService(session=session, message_service=msg_svc, session_service=session_svc)

        # 1. Identify target HR document
        stmt = select(Document).where(Document.deleted_at.is_(None)).order_by(Document.created_at.desc())
        docs = (await session.execute(stmt)).scalars().all()

        target_doc = next((d for d in docs if "hr" in d.title.lower() or "framework" in d.title.lower()), docs[0] if docs else None)
        if not target_doc:
            print("ERROR: No ready document found in database for testing.")
            return

        user_id = target_doc.user_id
        session_id = uuid.uuid4()

        # Create a mock chat session in DB
        from app.models.chat_session import ChatSession
        chat_sess = ChatSession(id=session_id, user_id=user_id, title="E2E Validation Session")
        session.add(chat_sess)
        await session.commit()

        print("\n=======================================================")
        print("PART 1: WHOLE DOCUMENT SUMMARY END-TO-END VALIDATION")
        print("=======================================================\n")

        summary_query = "Summarize the new HR framework document and tell me more detail"
        classified_route = classify(summary_query)
        print(f"QUERY: {summary_query}")
        print(f"DETECTED INTENT: {classified_route.value}")
        print(f"RESOLVED DOCUMENT: {target_doc.title} (ID: {target_doc.id})")

        filters = SearchFilters(user_id=user_id, document_id=target_doc.id)

        # Run ask_stream to capture telemetry and response
        stream_events = []
        full_response_text = ""
        sources = []

        async for event in rag_svc.ask_stream(session_id=session_id, question=summary_query, filters=filters):
            if "type': 'token'" in event or '"type": "token"' in event:
                import json
                try:
                    data = json.loads(event.replace("data: ", "").strip())
                    if data.get("type") == "token":
                        full_response_text += data.get("content", "")
                    elif data.get("type") == "meta":
                        sources = data.get("sources", [])
                except Exception:
                    pass

        sections_retrieved = list(set([s.get("section_title") for s in sources if s.get("section_title")]))
        print(f"NUMBER OF SECTIONS RETRIEVED: {len(sections_retrieved)}")
        print(f"SECTIONS LIST: {sections_retrieved}")
        print(f"NUMBER OF CHUNKS SENT TO LLM: {len(sources)}")
        print(f"FINAL CONTEXT SIZE: {sum(len(s.get('chunk_text', '')) for s in sources)} characters")

        print("\nEXACT FINAL LLM RESPONSE:")
        print("-------------------------------------------------------")
        print(full_response_text)
        print("-------------------------------------------------------")

        print("\n=======================================================")
        print("PART 2: 8 FACTUAL QUERIES END-TO-END VALIDATION")
        print("=======================================================\n")

        test_queries = [
            ("1. What are the working hours?", "working hours"),
            ("2. How many casual leaves are available?", "casual leave"),
            ("3. Can casual leave be carried forward?", "carried forward"),
            ("4. What is the WFH policy?", "wfh"),
            ("5. How does performance management work?", "performance"),
            ("6. What is the grievance process?", "grievance"),
            ("7. What does the document say about POSH?", "posh"),
            ("8. What happens during exit or termination?", "exit"),
        ]

        for q_text, expected_kw in test_queries:
            q_session_id = uuid.uuid4()
            q_sess = ChatSession(id=q_session_id, user_id=user_id, title="Factual Test Session")
            session.add(q_sess)
            await session.commit()

            q_route = classify(q_text)
            q_ans_text = ""
            q_sources = []

            async for event in rag_svc.ask_stream(session_id=q_session_id, question=q_text, filters=filters):
                import json
                try:
                    if event.startswith("data: "):
                        data = json.loads(event.replace("data: ", "").strip())
                        if data.get("type") == "token":
                            q_ans_text += data.get("content", "")
                        elif data.get("type") == "meta":
                            q_sources = data.get("sources", [])
                except Exception:
                    pass

            ret_secs = list(set([s.get("section_title") for s in q_sources if s.get("section_title")]))
            is_pass = len(q_sources) > 0 and len(q_ans_text.strip()) > 30 and ("not specify" not in q_ans_text.lower() or expected_kw in q_ans_text.lower())

            print(f"QUERY: {q_text}")
            print(f"INTENT: {q_route.value}")
            print(f"DOCUMENT: {target_doc.title}")
            print(f"RETRIEVED SECTION: {ret_secs}")
            print(f"RETRIEVED CHUNKS: {len(q_sources)}")
            print(f"FINAL ANSWER:\n{q_ans_text.strip()[:300]}...")
            print(f"PASS/FAIL: {'PASS' if is_pass else 'FAIL'}\n")
            print("-" * 55 + "\n")


if __name__ == "__main__":
    asyncio.run(run_e2e_validation())
