"""Unit test suite for DocumentParser structure preservation across DOCX, PDF, XLSX, TXT, and Markdown."""
from __future__ import annotations

import io
import uuid
import pytest
from app.services.parser import DocumentParser, ParsedDocument
from app.services.metadata import BlockType


class TestDocumentParserSuite:
    @pytest.fixture
    def parser(self) -> DocumentParser:
        return DocumentParser()

    def test_parse_markdown_blocks(self, parser: DocumentParser) -> None:
        md_text = (
            "# SipraHub Overview\n\n"
            "SipraHub is a high-performance local RAG platform.\n\n"
            "## Working Hours\n\n"
            "Working hours are Monday to Friday, 9:00 AM to 6:00 PM.\n\n"
            "| Day | Hours |\n"
            "| --- | --- |\n"
            "| Mon-Fri | 9:00 AM - 6:00 PM |\n"
        )
        content = md_text.encode("utf-8")
        doc_id = uuid.uuid4()
        parsed: ParsedDocument = parser.parse_sync(content, "SipraHub_Policy.md", doc_id)

        assert parsed.document_name == "SipraHub_Policy.md"
        assert len(parsed.blocks) >= 4

        types = [b.block_type for b in parsed.blocks]
        assert BlockType.HEADING in types
        assert BlockType.SUBHEADING in types
        assert BlockType.TABLE in types

        table_block = [b for b in parsed.blocks if b.block_type == BlockType.TABLE][0]
        assert "Mon-Fri" in table_block.text
        assert "9:00 AM - 6:00 PM" in table_block.text

    def test_parse_plain_text(self, parser: DocumentParser) -> None:
        txt = (
            "Company Policy\n\n"
            "Working Hours:\n"
            "Standard working hours for Sipra Hub are 9:00 AM to 6:00 PM.\n"
        )
        content = txt.encode("utf-8")
        doc_id = uuid.uuid4()
        parsed: ParsedDocument = parser.parse_sync(content, "Policy.txt", doc_id)

        assert len(parsed.blocks) >= 2
        full_text = "\n".join(b.text for b in parsed.blocks)
        assert "9:00 AM to 6:00 PM" in full_text

    def test_parse_empty_file_raises_error(self, parser: DocumentParser) -> None:
        from app.processing.parser import CorruptedFileError
        with pytest.raises(CorruptedFileError):
            parser.parse_sync(b"", "empty.txt", uuid.uuid4())

    def test_semantic_chunker_attaches_heading_context(self, parser: DocumentParser) -> None:
        from app.services.chunker import SemanticChunker
        md_text = (
            "# SipraHub Policy\n\n"
            "## Working Hours\n\n"
            "Standard operating hours are Monday to Friday, 9 AM to 6 PM.\n"
        )
        parsed = parser.parse_sync(md_text.encode("utf-8"), "SipraHub_Policy.md", uuid.uuid4())
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(parsed)

        assert len(chunks) >= 1
        first_chunk = chunks[0]
        assert "Working Hours" in first_chunk.text
        assert "9 AM to 6 PM" in first_chunk.text

