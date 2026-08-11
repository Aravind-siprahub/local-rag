import asyncio
import logging
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.services.ingestion_service import IngestionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process_all_pending")

async def process_all():
    async with AsyncSessionLocal() as session:
        stmt = select(Document).where(Document.deleted_at.is_(None))
        docs = list((await session.execute(stmt)).scalars().all())
        print(f"Total active documents found: {len(docs)}")
        ingestion = IngestionService(session)
        for doc in docs:
            print(f"Document: id={doc.id}, title={doc.title!r}, status={doc.status}")
            if doc.status in (DocumentStatus.UPLOADED, DocumentStatus.PROCESSING, "uploaded", "processing") or True:
                print(f"Processing document {doc.id} ({doc.title})...")
                try:
                    res = await ingestion.run_pipeline(doc.id)
                    await session.commit()
                    print(f"SUCCESS: {res}")
                except Exception as e:
                    import traceback
                    print(f"FAILED for {doc.id}: {e}")
                    traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(process_all())
