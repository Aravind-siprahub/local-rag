"""Reindex active documents with enhanced heading detection, hierarchy, and section breadcrumbs."""
import asyncio
import io
import logging
import os
import sys
import time
import uuid
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("reindex_documents")

from sqlalchemy import select, delete
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.embedding import Embedding
from app.models.enums import DocumentStatus, DocumentVersionStatus
from app.services.parser import DocumentParser
from app.services.chunker import chunk_document
from app.services.embedding import normalize_text_for_embedding
from app.embeddings.client import OllamaEmbeddingClient
from app.core.config import get_settings


async def reindex_document(session, doc: Document):
    settings = get_settings()
    upload_dir = Path(settings.UPLOAD_DIR)

    # Get current version
    stmt = (
        select(DocumentVersion)
        .where(DocumentVersion.document_id == doc.id)
        .order_by(DocumentVersion.version_number.desc())
    )
    res = await session.execute(stmt)
    version = res.scalars().first()

    if not version:
        logger.warning("No version found for document %s (%s)", doc.title, doc.id)
        return

    # Locate file on disk
    file_path = None
    if version.storage_path:
        candidate = upload_dir / version.storage_path.lstrip("/")
        if candidate.exists():
            file_path = candidate

    if not file_path:
        # Search upload_dir recursively for matching filename
        for p in upload_dir.rglob(f"*{Path(doc.title).name}*"):
            if p.is_file():
                file_path = p
                break

    if not file_path or not file_path.exists():
        logger.warning("Could not find file on disk for document %s (storage_path=%s)", doc.title, version.storage_path)
        return

    logger.info("Reindexing document %r (ID: %s) from %s", doc.title, doc.id, file_path)

    raw_bytes = file_path.read_bytes()
    parser = DocumentParser()
    parsed_doc = parser.parse_sync(raw_bytes, doc.title, doc.id, version.mime_type)

    heading_blocks = [b for b in parsed_doc.blocks if b.block_type.value in ('heading', 'subheading')]
    logger.info("Parsed %d blocks (%d headings detected)", len(parsed_doc.blocks), len(heading_blocks))

    chunks = chunk_document(parsed_doc)
    logger.info("Generated %d semantic chunks with hierarchy breadcrumbs", len(chunks))

    # Delete existing embeddings and chunks for this version
    chunk_ids_stmt = select(DocumentChunk.id).where(DocumentChunk.document_version_id == version.id)
    chunk_ids = (await session.execute(chunk_ids_stmt)).scalars().all()

    if chunk_ids:
        del_emb = delete(Embedding).where(Embedding.chunk_id.in_(chunk_ids))
        await session.execute(del_emb)

        del_chunks = delete(DocumentChunk).where(DocumentChunk.document_version_id == version.id)
        await session.execute(del_chunks)

    # Insert new chunks
    client = OllamaEmbeddingClient()
    model_name = settings.EMBEDDING_MODEL

    created_chunks = []
    for sc in chunks:
        norm_text = normalize_text_for_embedding(sc.text)
        new_chunk = DocumentChunk(
            id=sc.id if isinstance(sc.id, uuid.UUID) else uuid.UUID(str(sc.id)),
            document_version_id=version.id,
            chunk_index=sc.chunk_index,
            content=norm_text,
            content_tokens=sc.token_count,
            page_number=sc.page_number or None,
            section_title=sc.breadcrumb or sc.section or None,
            char_start=sc.char_start,
            char_end=sc.char_end,
            metadata_=sc.to_metadata_dict(),
        )
        session.add(new_chunk)
        created_chunks.append((new_chunk, norm_text))

    await session.flush()
    logger.info("Persisted %d chunks. Computing embeddings in batches...", len(created_chunks))

    # Compute embeddings in batches of 10
    batch_size = 10
    for i in range(0, len(created_chunks), batch_size):
        batch = created_chunks[i:i + batch_size]
        texts = [item[1] for item in batch]
        embed_tasks = [client.embed(t) for t in texts]
        embeddings = await asyncio.gather(*embed_tasks)
        for (chunk_obj, _), emb_vector in zip(batch, embeddings):
            emb_record = Embedding(
                chunk_id=chunk_obj.id,
                embedding=emb_vector,
                model_name=model_name,
                dimensions=len(emb_vector),
            )
            session.add(emb_record)

    doc.status = DocumentStatus.READY
    doc.current_version_id = version.id
    version.status = DocumentVersionStatus.COMPLETED

    await session.commit()
    logger.info("Successfully reindexed %r (ID: %s, Version: %s) with %d chunks!", doc.title, doc.id, version.id, len(created_chunks))


async def main():
    async with AsyncSessionLocal() as session:
        # Reindex HR Framework documents
        stmt = (
            select(Document)
            .where(Document.deleted_at.is_(None))
            .where(Document.title.ilike("%HR Framework%"))
            .order_by(Document.created_at.desc())
        )
        res = await session.execute(stmt)
        docs = res.scalars().all()

        print(f"Found {len(docs)} matching document(s) for reindexing.")
        for d in docs:
            await reindex_document(session, d)

        print("\nAll target documents reindexed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
