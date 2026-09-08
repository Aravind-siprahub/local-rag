"""Semantic document chunker — hierarchy-aware, token-based splitting."""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from typing import Iterator

from app.core.config import get_settings
from app.services.keyword_extractor import extract_keywords
from app.services.metadata import (
    BlockType,
    Chunk,
    ChunkingConfig,
    ContentType,
    DocumentBlock,
    HierarchyContext,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

_HEADING_ONLY_RE = re.compile(r"^#{1,6}\s+\S")
_PAGE_NUM_ONLY_RE = re.compile(r"^\s*\d{1,4}\s*$")


class SemanticChunker:
    """Convert a ParsedDocument into validated semantic chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        if config is None:
            settings = get_settings()
            config = ChunkingConfig(
                min_tokens=settings.SEMANTIC_CHUNK_MIN_TOKENS,
                max_tokens=settings.SEMANTIC_CHUNK_MAX_TOKENS,
                overlap_min=settings.SEMANTIC_CHUNK_OVERLAP_MIN,
                overlap_max=settings.SEMANTIC_CHUNK_OVERLAP_MAX,
                min_meaningful_chars=settings.SEMANTIC_CHUNK_MIN_CHARS,
            )
        self.config = config
        self._token_counter = TokenCounter()

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Primary entry point: parse structured document → semantic chunks."""
        hierarchy = HierarchyContext()
        raw_chunks: list[Chunk] = []
        char_offset = 0

        for block in document.blocks:
            hierarchy = self._update_hierarchy(hierarchy, block)
            block_chunks = self._chunk_block(
                block, document, hierarchy, char_offset
            )
            raw_chunks.extend(block_chunks)
            char_offset += len(block.text) + 1

        validated = self._validate_chunks(raw_chunks)
        deduped = self._deduplicate_chunks(validated)

        # Assign final indices and keyword extraction.
        total = len(deduped)
        final: list[Chunk] = []
        for idx, chunk in enumerate(deduped):
            keywords = extract_keywords(chunk.text, language=chunk.language)
            final.append(chunk.model_copy(update={
                "chunk_index": idx,
                "total_chunks": total,
                "keywords": keywords,
                "token_count": self._token_counter.count(chunk.text),
            }))

        logger.info(
            "Chunked document %s: %d blocks → %d chunks (parser=%s)",
            document.document_name,
            len(document.blocks),
            len(final),
            document.parser_used,
        )
        return final

    def _update_hierarchy(
        self, hierarchy: HierarchyContext, block: DocumentBlock
    ) -> HierarchyContext:
        if block.block_type in (BlockType.HEADING, BlockType.SUBHEADING):
            level = block.level or (1 if block.block_type == BlockType.HEADING else 2)
            return hierarchy.with_heading(block.text, level)
        return hierarchy

    def _chunk_block(
        self,
        block: DocumentBlock,
        document: ParsedDocument,
        hierarchy: HierarchyContext,
        char_offset: int,
    ) -> list[Chunk]:
        content_type = self._block_to_content_type(block.block_type)
        block_meta = getattr(block, "metadata", None) or {}

        # Atomic blocks — never split.
        if block.block_type in (
            BlockType.TABLE, BlockType.LIST, BlockType.FAQ,
            BlockType.CODE, BlockType.IMAGE_CAPTION,
        ):
            return [self._make_chunk(
                text=block.text,
                document=document,
                hierarchy=hierarchy,
                content_type=content_type,
                page_number=block.page_number or 0,
                char_start=char_offset,
                char_end=char_offset + len(block.text),
                metadata=block_meta,
            )]

        # Headings become context only — not standalone chunks unless they carry body text.
        if block.block_type in (BlockType.HEADING, BlockType.SUBHEADING):
            return []

        # Paragraphs: split on token budget if needed.
        token_count = self._token_counter.count(block.text)
        if token_count <= self.config.max_tokens:
            return [self._make_chunk(
                text=block.text,
                document=document,
                hierarchy=hierarchy,
                content_type=content_type,
                page_number=block.page_number or 0,
                char_start=char_offset,
                char_end=char_offset + len(block.text),
                metadata=block_meta,
            )]

        return self._split_paragraph_block(
            block.text, document, hierarchy, content_type,
            block.page_number or 0, char_offset,
            metadata=block_meta,
        )

    def _split_paragraph_block(
        self,
        text: str,
        document: ParsedDocument,
        hierarchy: HierarchyContext,
        content_type: ContentType,
        page_number: int,
        char_offset: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split oversized paragraphs on paragraph/sentence boundaries with overlap."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        part_start = char_offset

        for para in paragraphs:
            para_tokens = self._token_counter.count(para)

            # Single paragraph exceeds max — split on sentences.
            if para_tokens > self.config.max_tokens:
                if current_parts:
                    chunk_text = "\n\n".join(current_parts)
                    chunks.append(self._make_chunk(
                        text=chunk_text, document=document, hierarchy=hierarchy,
                        content_type=content_type, page_number=page_number,
                        char_start=part_start, char_end=part_start + len(chunk_text),
                        metadata=metadata,
                    ))
                    current_parts = []
                    current_tokens = 0

                sentence_chunks = self._split_on_sentences(
                    para, document, hierarchy, content_type, page_number, char_offset,
                    metadata=metadata,
                )
                chunks.extend(sentence_chunks)
                part_start = char_offset + len(para)
                continue

            if current_tokens + para_tokens > self.config.max_tokens and current_parts:
                chunk_text = "\n\n".join(current_parts)
                chunks.append(self._make_chunk(
                    text=chunk_text, document=document, hierarchy=hierarchy,
                    content_type=content_type, page_number=page_number,
                    char_start=part_start, char_end=part_start + len(chunk_text),
                    metadata=metadata,
                ))
                # Overlap: carry trailing paragraph(s) within overlap budget.
                overlap_parts = self._compute_overlap(current_parts)
                current_parts = overlap_parts
                current_tokens = sum(self._token_counter.count(p) for p in current_parts)
                part_start = char_offset

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(self._make_chunk(
                text=chunk_text, document=document, hierarchy=hierarchy,
                content_type=content_type, page_number=page_number,
                char_start=part_start, char_end=part_start + len(chunk_text),
                metadata=metadata,
            ))

        return chunks

    def _split_on_sentences(
        self,
        text: str,
        document: ParsedDocument,
        hierarchy: HierarchyContext,
        content_type: ContentType,
        page_number: int,
        char_offset: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Last-resort split on sentence boundaries (never mid-sentence)."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        part_start = char_offset

        for sentence in sentences:
            sent_tokens = self._token_counter.count(sentence)
            if current_tokens + sent_tokens > self.config.max_tokens and current:
                chunk_text = " ".join(current)
                chunks.append(self._make_chunk(
                    text=chunk_text, document=document, hierarchy=hierarchy,
                    content_type=content_type, page_number=page_number,
                    char_start=part_start, char_end=part_start + len(chunk_text),
                    metadata=metadata,
                ))
                overlap = self._compute_overlap(current)
                current = overlap
                current_tokens = sum(self._token_counter.count(s) for s in current)
                part_start = char_offset

            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            chunk_text = " ".join(current)
            chunks.append(self._make_chunk(
                text=chunk_text, document=document, hierarchy=hierarchy,
                content_type=content_type, page_number=page_number,
                char_start=part_start, char_end=part_start + len(chunk_text),
                metadata=metadata,
            ))

        return chunks

    def _compute_overlap(self, parts: list[str]) -> list[str]:
        """Select trailing parts that fit within the overlap token budget."""
        if not parts:
            return []
        overlap_budget = self.config.overlap_max
        selected: list[str] = []
        tokens_used = 0
        for part in reversed(parts):
            part_tokens = self._token_counter.count(part)
            if tokens_used + part_tokens > overlap_budget:
                break
            selected.insert(0, part)
            tokens_used += part_tokens
            if tokens_used >= self.config.overlap_min:
                break
        return selected

    def _make_chunk(
        self,
        *,
        text: str,
        document: ParsedDocument,
        hierarchy: HierarchyContext,
        content_type: ContentType,
        page_number: int,
        char_start: int,
        char_end: int,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        heading_prefix = ""
        if hierarchy.breadcrumb and not text.lower().startswith(hierarchy.breadcrumb.lower()):
            heading_prefix = f"[{hierarchy.breadcrumb}]\n"
        elif hierarchy.section and not text.lower().startswith(hierarchy.section.lower()):
            heading_prefix = f"[{hierarchy.section}]\n"

        chunk_text = f"{heading_prefix}{text}" if heading_prefix else text
        chunk_id = self._generate_chunk_id(document.document_id, chunk_text, char_start)

        ext = (document.source_format or "").lower().strip(".")
        filename = document.document_name
        source_loc = ""

        # Extract sheet name and rows if available
        sheet = None
        r_start = None
        r_end = None
        if metadata:
            sheet = metadata.get("sheet")
            r_start = metadata.get("row_start")
            r_end = metadata.get("row_end")
        if not sheet:
            sheet_m = re.search(r"(?i)\bSheet:\s*([^\n\r,\)]+)", text)
            if sheet_m:
                sheet = sheet_m.group(1).strip()

        if ext in ("xlsx", "xls", "csv") or sheet:
            if sheet:
                if r_start is not None and r_end is not None:
                    source_loc = f"Sheet: {sheet}, Rows {r_start}-{r_end}"
                else:
                    source_loc = f"Sheet: {sheet}"
        elif ext == "pdf":
            if page_number and page_number > 0:
                source_loc = f"p. {page_number}"
        elif ext in ("docx", "doc"):
            if page_number and page_number > 0:
                source_loc = f"p. {page_number}"
            elif hierarchy.section:
                source_loc = f"Section: {hierarchy.section}"
            elif hierarchy.breadcrumb:
                source_loc = f"Section: {hierarchy.breadcrumb}"
        elif ext in ("md", "markdown"):
            if hierarchy.section:
                source_loc = hierarchy.section
            elif hierarchy.breadcrumb:
                source_loc = hierarchy.breadcrumb
        else:
            if page_number and page_number > 0:
                source_loc = f"p. {page_number}"
            elif hierarchy.section:
                source_loc = f"Section: {hierarchy.section}"
            elif hierarchy.breadcrumb:
                source_loc = f"Section: {hierarchy.breadcrumb}"

        return Chunk(
            id=chunk_id,
            document_id=document.document_id,
            document_name=document.document_name,
            file_name=filename,
            file_type=ext,
            source_location=source_loc,
            page_number=page_number,
            section=hierarchy.section,
            subsection=hierarchy.subsection,
            breadcrumb=hierarchy.breadcrumb,
            content_type=content_type,
            language=document.language,
            text=chunk_text,
            char_start=char_start,
            char_end=char_end,
            token_count=self._token_counter.count(chunk_text),
        )

    def _validate_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Reject invalid chunks per validation rules."""
        valid: list[Chunk] = []
        for chunk in chunks:
            if self._is_valid_chunk(chunk):
                valid.append(chunk)
            else:
                logger.debug(
                    "Rejected chunk (index pending): type=%s, chars=%d, preview=%r",
                    chunk.content_type.value,
                    len(chunk.text),
                    chunk.text[:60],
                )
        return valid

    def _is_valid_chunk(self, chunk: Chunk) -> bool:
        text = chunk.text.strip()
        if not text:
            return False
        if _PAGE_NUM_ONLY_RE.match(text):
            return False
        if _HEADING_ONLY_RE.match(text) and chunk.content_type == ContentType.PARAGRAPH:
            return False
        meaningful = re.sub(r"\s+", "", text)
        if len(meaningful) < self.config.min_meaningful_chars:
            # Allow shorter atomic types (FAQ, code snippets, lists, tables) or items under an active section/breadcrumb
            if chunk.content_type in (ContentType.CODE, ContentType.FAQ, ContentType.LIST, ContentType.TABLE):
                pass
            elif (chunk.breadcrumb or chunk.section) and len(meaningful) >= 15:
                pass
            else:
                return False
        # Mostly whitespace check.
        if len(meaningful) / max(len(text), 1) < 0.3:
            return False
        return True

    def _deduplicate_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Remove exact and near-duplicate chunks."""
        seen_hashes: set[str] = set()
        unique: list[Chunk] = []
        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", chunk.text.strip().lower())
            content_hash = hashlib.sha256(normalized.encode()).hexdigest()
            if content_hash in seen_hashes:
                logger.debug("Skipping duplicate chunk: %r", chunk.text[:50])
                continue
            seen_hashes.add(content_hash)
            unique.append(chunk)
        return unique

    @staticmethod
    def _block_to_content_type(block_type: BlockType) -> ContentType:
        mapping = {
            BlockType.TABLE: ContentType.TABLE,
            BlockType.CODE: ContentType.CODE,
            BlockType.FAQ: ContentType.FAQ,
            BlockType.LIST: ContentType.LIST,
            BlockType.IMAGE_CAPTION: ContentType.IMAGE_CAPTION,
        }
        return mapping.get(block_type, ContentType.PARAGRAPH)

    @staticmethod
    def _generate_chunk_id(document_id: uuid.UUID, text: str, offset: int) -> str:
        payload = f"{document_id}:{offset}:{text[:100]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


class TokenCounter:
    """Token counting with tiktoken fallback to character heuristic."""

    def __init__(self) -> None:
        self._encoding = None
        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        # Heuristic: ~4 chars per token for English prose.
        return max(1, len(text) // 4)


# Module-level convenience.
_default_chunker: SemanticChunker | None = None


def get_semantic_chunker(config: ChunkingConfig | None = None) -> SemanticChunker:
    global _default_chunker
    if config is not None:
        return SemanticChunker(config)
    if _default_chunker is None:
        _default_chunker = SemanticChunker()
    return _default_chunker


def chunk_document(document: ParsedDocument, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Convenience function matching the required API signature."""
    return get_semantic_chunker(config).chunk_document(document)
