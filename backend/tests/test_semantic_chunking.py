"""Unit tests for semantic chunking pipeline services."""
from __future__ import annotations

import uuid

import pytest

from app.services.chunker import SemanticChunker, chunk_document
from app.services.embedding import normalize_text_for_embedding, prepare_chunk_for_embedding
from app.services.keyword_extractor import extract_keywords
from app.services.metadata import (
    BlockType,
    ChunkingConfig,
    ContentType,
    DocumentBlock,
    HierarchyContext,
    ParsedDocument,
)
from app.services.parser import DocumentParser


DOC_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _make_doc(blocks: list[DocumentBlock], **kwargs) -> ParsedDocument:
    return ParsedDocument(
        document_id=DOC_ID,
        document_name="test.md",
        blocks=blocks,
        **kwargs,
    )


class TestHierarchyContext:
    def test_breadcrumb_for_nested_headings(self) -> None:
        ctx = HierarchyContext()
        ctx = ctx.with_heading("HR", 1)
        ctx = ctx.with_heading("Leave", 2)
        ctx = ctx.with_heading("Annual Leave", 3)
        assert ctx.breadcrumb == "HR → Leave → Annual Leave"
        assert ctx.section == "HR"
        assert ctx.subsection == "Leave"


class TestKeywordExtractor:
    def test_extracts_keywords_from_auth_text(self) -> None:
        text = (
            "Authentication uses JWT tokens. The refresh token and access token "
            "provide secure session management for API security."
        )
        keywords = extract_keywords(text)
        assert 5 <= len(keywords) <= 15
        assert any("token" in kw or "jwt" in kw or "authentication" in kw for kw in keywords)

    def test_returns_empty_for_blank(self) -> None:
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []


class TestEmbeddingPrep:
    def test_normalizes_whitespace_preserves_paragraphs(self) -> None:
        raw = "Hello   world.\n\nSecond   paragraph."
        result = normalize_text_for_embedding(raw)
        assert "Hello world." in result
        assert "Second paragraph." in result
        assert "\n\n" in result

    def test_preserves_code_fences(self) -> None:
        raw = "Intro\n\n```python\ndef foo():\n    pass\n```\n\nOutro"
        result = normalize_text_for_embedding(raw)
        assert "```python" in result
        assert "def foo():" in result

    def test_prepare_includes_breadcrumb(self) -> None:
        from app.services.metadata import Chunk

        chunk = Chunk(
            id="abc",
            document_id=DOC_ID,
            document_name="doc.md",
            breadcrumb="HR → Leave",
            text="Employees receive 20 days annual leave.",
        )
        prepared = prepare_chunk_for_embedding(chunk)
        assert "[HR → Leave]" in prepared
        assert "20 days" in prepared


class TestSemanticChunker:
    def _chunker(self) -> SemanticChunker:
        return SemanticChunker(ChunkingConfig(
            min_tokens=10,
            max_tokens=50,
            overlap_min=5,
            overlap_max=10,
            min_meaningful_chars=20,
        ))

    def test_single_paragraph_produces_one_chunk(self) -> None:
        doc = _make_doc([
            DocumentBlock(block_type=BlockType.HEADING, text="Introduction", level=1),
            DocumentBlock(
                block_type=BlockType.PARAGRAPH,
                text="This is a coherent paragraph about machine learning applications in healthcare.",
            ),
        ])
        chunks = self._chunker().chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].section == "Introduction"
        assert chunks[0].breadcrumb == "Introduction"
        assert chunks[0].content_type == ContentType.PARAGRAPH

    def test_table_stays_atomic(self) -> None:
        table = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        doc = _make_doc([DocumentBlock(block_type=BlockType.TABLE, text=table)])
        chunks = self._chunker().chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].content_type == ContentType.TABLE
        assert "Alice" in chunks[0].text and "Bob" in chunks[0].text

    def test_list_stays_atomic(self) -> None:
        lst = "- First item about databases\n- Second item about caching\n- Third item about APIs"
        doc = _make_doc([DocumentBlock(block_type=BlockType.LIST, text=lst)])
        chunks = self._chunker().chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].content_type == ContentType.LIST

    def test_faq_one_chunk_per_pair(self) -> None:
        doc = _make_doc([
            DocumentBlock(
                block_type=BlockType.FAQ,
                text="Q: What is the refund policy?\nA: Refunds are processed within 14 business days.",
            ),
            DocumentBlock(
                block_type=BlockType.FAQ,
                text="Q: How do I cancel?\nA: Contact support via email to cancel your subscription.",
            ),
        ])
        chunks = self._chunker().chunk_document(doc)
        assert len(chunks) == 2
        assert all(c.content_type == ContentType.FAQ for c in chunks)

    def test_code_not_split_inside_function(self) -> None:
        code = "def authenticate(user, password):\n    token = generate_jwt(user)\n    return token"
        doc = _make_doc([DocumentBlock(block_type=BlockType.CODE, text=code)])
        chunks = self._chunker().chunk_document(doc)
        assert len(chunks) == 1
        assert "def authenticate" in chunks[0].text

    def test_long_paragraph_splits_on_boundaries(self) -> None:
        paragraphs = [
            "This is sentence one about distributed systems and consensus protocols in detail. " * 3,
            "This is sentence two about replication strategies and fault tolerance mechanisms. " * 3,
            "This is sentence three about leader election and log compaction in production. " * 3,
        ]
        doc = _make_doc([
            DocumentBlock(block_type=BlockType.PARAGRAPH, text="\n\n".join(paragraphs)),
        ])
        chunks = self._chunker().chunk_document(doc)
        assert len(chunks) >= 2

    def test_rejects_heading_only_chunks(self) -> None:
        doc = _make_doc([
            DocumentBlock(block_type=BlockType.PARAGRAPH, text="Short"),
        ])
        chunks = SemanticChunker(ChunkingConfig(min_meaningful_chars=50)).chunk_document(doc)
        assert len(chunks) == 0

    def test_rejects_duplicate_chunks(self) -> None:
        text = "Identical content about cloud infrastructure and deployment pipelines in production."
        doc = _make_doc([
            DocumentBlock(block_type=BlockType.PARAGRAPH, text=text),
            DocumentBlock(block_type=BlockType.PARAGRAPH, text=text),
        ])
        chunks = self._chunker().chunk_document(doc)
        assert len(chunks) == 1

    def test_nested_headings_set_breadcrumb(self) -> None:
        doc = _make_doc([
            DocumentBlock(block_type=BlockType.HEADING, text="HR", level=1),
            DocumentBlock(block_type=BlockType.SUBHEADING, text="Leave", level=2),
            DocumentBlock(
                block_type=BlockType.PARAGRAPH,
                text="Annual leave policy grants employees twenty working days per calendar year.",
            ),
        ])
        chunks = self._chunker().chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].breadcrumb == "HR → Leave"

    def test_chunk_metadata_schema(self) -> None:
        doc = _make_doc([
            DocumentBlock(
                block_type=BlockType.PARAGRAPH,
                text="Vector databases enable semantic search across large document collections efficiently.",
            ),
        ])
        chunks = self._chunker().chunk_document(doc)
        meta = chunks[0].to_metadata_dict()
        required_keys = {
            "id", "document_id", "document_name", "page_number", "section",
            "subsection", "breadcrumb", "chunk_index", "total_chunks",
            "token_count", "content_type", "keywords", "language",
        }
        assert required_keys.issubset(meta.keys())
        pg_record = chunks[0].to_pgvector_record()
        assert "chunk_id" in pg_record
        assert "embedding" in pg_record
        assert "metadata" in pg_record
        assert "text" in pg_record

    def test_keywords_populated(self) -> None:
        doc = _make_doc([
            DocumentBlock(
                block_type=BlockType.PARAGRAPH,
                text="JWT authentication uses access tokens and refresh tokens for secure API access.",
            ),
        ])
        chunks = self._chunker().chunk_document(doc)
        assert 5 <= len(chunks[0].keywords) <= 15


class TestDocumentParser:
    def setup_method(self) -> None:
        self.parser = DocumentParser()

    def test_parse_markdown_with_headings_and_code(self) -> None:
        md = b"""# API Guide

## Authentication

Use JWT tokens for all requests.

```python
def get_token():
    return "abc"
```

## Endpoints

- GET /users
- POST /users
"""
        doc = self.parser.parse_sync(md, "guide.md", DOC_ID)
        block_types = [b.block_type for b in doc.blocks]
        assert BlockType.HEADING in block_types
        assert BlockType.CODE in block_types
        assert BlockType.LIST in block_types

    def test_parse_markdown_table(self) -> None:
        md = b"""# Data

| Col1 | Col2 |
|------|------|
| A    | B    |
| C    | D    |
"""
        doc = self.parser.parse_sync(md, "table.md", DOC_ID)
        tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
        assert len(tables) == 1
        assert "Col1" in tables[0].text

    def test_parse_faq_content(self) -> None:
        text = b"""Q: What is RAG?
A: Retrieval Augmented Generation combines search with language models.

Q: Why use it?
A: It grounds answers in your own documents.
"""
        doc = self.parser.parse_sync(text, "faq.txt", DOC_ID)
        faqs = [b for b in doc.blocks if b.block_type == BlockType.FAQ]
        assert len(faqs) == 2

    def test_parse_multilingual_cjk(self) -> None:
        text = "人工智能和机器学习正在改变软件开发的方式。".encode()
        doc = self.parser.parse_sync(text, "zh.txt", DOC_ID)
        assert doc.language == "zh"
        assert len(doc.blocks) >= 1

    def test_filters_page_numbers(self) -> None:
        text = b"Real content about software engineering best practices here.\n\n42\n\nMore content follows."
        doc = self.parser.parse_sync(text, "pages.txt", DOC_ID)
        for block in doc.blocks:
            assert block.text.strip() != "42"

    def test_parse_html_structure(self) -> None:
        html = b"""<html><body>
<h1>Title</h1>
<p>Paragraph about databases.</p>
<ul><li>Item one</li><li>Item two</li></ul>
</body></html>"""
        doc = self.parser.parse_sync(html, "page.html", DOC_ID)
        assert any(b.block_type == BlockType.HEADING for b in doc.blocks)
        assert any(b.block_type == BlockType.PARAGRAPH for b in doc.blocks)

    def test_rejects_empty_file(self) -> None:
        from app.services.parser import CorruptedFileError
        with pytest.raises(CorruptedFileError):
            self.parser.parse_sync(b"", "empty.txt", DOC_ID)

    def test_unsupported_format(self) -> None:
        from app.services.parser import UnsupportedFormatError
        with pytest.raises(UnsupportedFormatError):
            self.parser.parse_sync(b"data", "file.exe", DOC_ID)


class TestEndToEndPipeline:
    def test_chunk_document_convenience_function(self) -> None:
        doc = _make_doc([
            DocumentBlock(block_type=BlockType.HEADING, text="Security", level=1),
            DocumentBlock(
                block_type=BlockType.PARAGRAPH,
                text="OAuth 2.0 authorization framework enables secure delegated access to resources.",
            ),
        ])
        chunks = chunk_document(doc, ChunkingConfig(min_tokens=10, min_meaningful_chars=10, max_tokens=200))
        assert len(chunks) >= 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == len(chunks)
