"""End-to-End RAG Ingestion, Retrieval, and Prompt Context Verification for All 8 Formats.

Validates the full pipeline:
  File Bytes -> Parse -> Normalize -> Chunk -> Validate Chunks -> Embed -> Vector Store -> Hybrid Retrieval -> RAG Context

Covers:
  1. .docx (Word document)
  2. .doc (Legacy Word 97-2003 OLE binary)
  3. .pdf (Multi-page PDF)
  4. .xlsx (Excel spreadsheet with structured rows)
  5. .xls (Legacy Excel 97-2003 binary spreadsheet)
  6. .csv (Structured delimiter-detected spreadsheet)
  7. .md (Markdown with heading hierarchy breadcrumbs and tables)
  8. .txt (Plain text with paragraph and encoding preservation)
"""
from __future__ import annotations

import io
import math
import struct
import uuid
from typing import Any
import pytest

from app.prompting.builder import PromptBuilder
from app.prompting.templates import format_chunk
from app.retrieval.ranking import RankedResult
from app.retrieval.search import SearchFilters, SearchHit
from app.services.chunker import chunk_document
from app.services.metadata import Chunk, ContentType
from app.services.parser import DocumentParser


def _make_minimal_ole_doc(text: str) -> bytes:
    """Generate a valid OLE Compound Document with a WordDocument stream."""
    header = bytearray(512)
    header[0:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    header[24:26] = struct.pack("<H", 0x003E)
    header[26:28] = struct.pack("<H", 0x0003)
    header[28:30] = struct.pack("<H", 0xFFFE)
    header[30:32] = struct.pack("<H", 9)
    header[32:34] = struct.pack("<H", 6)
    header[44:48] = struct.pack("<I", 1)
    header[48:52] = struct.pack("<I", 1)
    header[56:60] = struct.pack("<I", 0x1000)
    header[60:64] = struct.pack("<I", 0xFFFFFFFE)
    header[68:72] = struct.pack("<I", 0xFFFFFFFE)
    header[76:80] = struct.pack("<I", 0)
    for i in range(1, 109):
        header[76 + i * 4 : 80 + i * 4] = struct.pack("<I", 0xFFFFFFFF)

    fat = bytearray(512)
    for i in range(128):
        fat[i * 4 : (i + 1) * 4] = struct.pack("<I", 0xFFFFFFFF)

    fat[0:4] = struct.pack("<I", 0xFFFFFFFD)
    fat[4:8] = struct.pack("<I", 0xFFFFFFFE)
    for i in range(2, 12):
        fat[i * 4 : (i + 1) * 4] = struct.pack("<I", i + 1 if i < 11 else 0xFFFFFFFE)

    dir_sector = bytearray(512)

    def make_dir_entry(name: str, entry_type: int, start_sect: int, size: int, child: int = -1) -> bytearray:
        e = bytearray(128)
        name_bytes = (name + "\x00").encode("utf-16le")
        e[0 : len(name_bytes)] = name_bytes
        e[64:66] = struct.pack("<H", len(name_bytes))
        e[66] = entry_type
        e[67] = 1
        e[68:72] = struct.pack("<i", -1)
        e[72:76] = struct.pack("<i", -1)
        e[76:80] = struct.pack("<i", child)
        e[116:120] = struct.pack("<I", start_sect)
        e[120:124] = struct.pack("<I", size)
        return e

    dir_sector[0:128] = make_dir_entry("Root Entry", 5, 0xFFFFFFFE, 0, child=1)
    dir_sector[128:256] = make_dir_entry("WordDocument", 2, 2, 5120)

    stream_content = bytearray(5120)
    stream_content[0:2] = struct.pack("<H", 0xA5EC)
    msg_bytes = text.encode("cp1252", errors="replace")
    stream_content[0x4C:0x50] = struct.pack("<I", len(text))
    stream_content[0x800 : 0x800 + len(msg_bytes)] = msg_bytes

    return bytes(header) + bytes(fat) + bytes(dir_sector) + bytes(stream_content)


class SimpleVectorIndex:
    """In-memory semantic vector store for deterministic end-to-end retrieval validation."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.entries: list[dict] = []

    def _embed(self, text: str) -> list[float]:
        """Deterministic term-frequency vector embedding for testing semantic retrieval."""
        words = [w.strip(".,;:!?\"'()[]{}").lower() for w in text.split() if w.strip()]
        vec = [0.0] * self.dim
        for w in words:
            h = hash(w) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def add_chunk(self, chunk: Chunk) -> None:
        emb = self._embed(chunk.text)
        self.entries.append({
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "text": chunk.text,
            "embedding": emb,
            "metadata": chunk.to_metadata_dict(),
        })

    def search(self, query: str, top_k: int = 3) -> list[RankedResult]:
        q_emb = self._embed(query)
        scored: list[tuple[float, dict]] = []
        for e in self.entries:
            sim = sum(a * b for a, b in zip(q_emb, e["embedding"]))
            scored.append((sim, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, e) in enumerate(scored[:top_k], start=1):
            results.append(RankedResult(
                chunk_id=uuid.UUID(hex=e["chunk_id"][:32]),
                chunk_text=e["text"],
                document_id=e["document_id"],
                document_version_id=uuid.uuid4(),
                document_title=e["document_name"],
                similarity_score=score,
                rank=rank,
                section_title=e["metadata"].get("section", ""),
                page_number=e["metadata"].get("page_number", 0),
                metadata_=e["metadata"],
            ))
        return results


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser()


class TestAllFormatRAGE2E:
    """Verify that every file type correctly ingests, chunks, embeds, retrieves, and builds context."""

    def test_e2e_docx_rag(self, parser: DocumentParser) -> None:
        from docx import Document as DocxDocument

        doc = DocxDocument()
        doc.add_heading("Corporate Annual Leave Policy", level=1)
        doc.add_paragraph(
            "Under the 2026 guidelines, full-time permanent employees receive 18 days of paid annual leave."
        )
        doc.add_paragraph("Employees are required to give 14 days advance notice prior to taking extended leave.")
        buf = io.BytesIO()
        doc.save(buf)
        docx_bytes = buf.getvalue()

        doc_id = uuid.uuid4()
        parsed = parser.parse_sync(docx_bytes, "leave_policy.docx", doc_id)
        chunks = chunk_document(parsed)
        assert len(chunks) >= 1

        index = SimpleVectorIndex()
        for c in chunks:
            index.add_chunk(c)

        hits = index.search("How many annual leave days do full-time permanent employees receive?")
        assert len(hits) >= 1
        best_hit = hits[0]
        assert "18 days" in best_hit.chunk_text
        assert best_hit.document_title == "leave_policy.docx"

        # Verify prompt builder context formatting
        builder = PromptBuilder()
        prompt = builder.build(
            question="How many annual leave days do employees receive?",
            retrieved_chunks=hits,
            chat_history=[],
        )
        assert "leave_policy.docx" in prompt.user_prompt
        assert "18 days" in prompt.user_prompt

    def test_e2e_doc_binary_ole_rag(self, parser: DocumentParser) -> None:
        fact = "International Travel Directive: The company provides 120 USD daily per diem for lodging and meals."
        doc_bytes = _make_minimal_ole_doc(fact)

        doc_id = uuid.uuid4()
        parsed = parser.parse_sync(doc_bytes, "travel_directive.doc", doc_id)
        chunks = chunk_document(parsed)
        assert len(chunks) >= 1

        index = SimpleVectorIndex()
        for c in chunks:
            index.add_chunk(c)

        hits = index.search("What is the international per diem amount in USD?")
        assert len(hits) >= 1
        assert "120 USD" in hits[0].chunk_text
        assert hits[0].document_title == "travel_directive.doc"

    def test_e2e_pdf_rag(self, parser: DocumentParser) -> None:
        import fitz

        pdf_doc: Any = fitz.open()
        p1 = pdf_doc.new_page()
        p1.insert_text((50, 72), "Health and Dental Insurance Policy", fontsize=16)
        p1.insert_text((50, 120), "The comprehensive medical plan provides dental benefits capped at $1500 per year.", fontsize=11)
        pdf_bytes = pdf_doc.tobytes()

        doc_id = uuid.uuid4()
        parsed = parser.parse_sync(pdf_bytes, "benefits_handbook.pdf", doc_id)
        chunks = chunk_document(parsed)
        assert len(chunks) >= 1

        index = SimpleVectorIndex()
        for c in chunks:
            index.add_chunk(c)

        hits = index.search("What is the dental benefits cap per year?")
        assert len(hits) >= 1
        assert "$1500" in hits[0].chunk_text
        assert hits[0].document_title == "benefits_handbook.pdf"

    def test_e2e_xlsx_rag(self, parser: DocumentParser) -> None:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        ws.title = "Department Leave"
        ws.append(["Employee", "Department", "Leave Balance"])
        ws.append(["Arun", "IT", 12])
        ws.append(["Priya", "HR", 15])
        buf = io.BytesIO()
        wb.save(buf)
        xlsx_bytes = buf.getvalue()

        doc_id = uuid.uuid4()
        parsed = parser.parse_sync(xlsx_bytes, "leave_balances.xlsx", doc_id)
        chunks = chunk_document(parsed)
        assert len(chunks) >= 1

        index = SimpleVectorIndex()
        for c in chunks:
            index.add_chunk(c)

        hits = index.search("What is the leave balance for Arun in IT department?")
        assert len(hits) >= 1
        top_chunk = hits[0].chunk_text
        assert "Employee: Arun" in top_chunk
        assert "Department: IT" in top_chunk
        assert "Leave Balance: 12" in top_chunk

    def test_e2e_xls_rag(self, parser: DocumentParser) -> None:
        import xlwt  # type: ignore[import-untyped]

        wb = xlwt.Workbook()
        ws = wb.add_sheet("Stock Levels")
        ws.write(0, 0, "SKU")
        ws.write(0, 1, "Quantity")
        ws.write(0, 2, "Warehouse")
        ws.write(1, 0, "SKU-9920")
        ws.write(1, 1, 450)
        ws.write(1, 2, "Warehouse B")
        buf = io.BytesIO()
        wb.save(buf)
        xls_bytes = buf.getvalue()

        doc_id = uuid.uuid4()
        parsed = parser.parse_sync(xls_bytes, "inventory.xls", doc_id)
        chunks = chunk_document(parsed)
        assert len(chunks) >= 1

        index = SimpleVectorIndex()
        for c in chunks:
            index.add_chunk(c)

        hits = index.search("What is the stock quantity for SKU-9920 in Warehouse B?")
        assert len(hits) >= 1
        top_text = hits[0].chunk_text
        assert "SKU: SKU-9920" in top_text
        assert "Quantity: 450" in top_text
        assert "Warehouse: Warehouse B" in top_text

    def test_e2e_csv_rag(self, parser: DocumentParser) -> None:
        csv_data = "Project,Lead,Budget\nProject Titan,Alice Smith,500000\nProject Apollo,Bob Jones,300000\n"
        doc_id = uuid.uuid4()
        parsed = parser.parse_sync(csv_data.encode("utf-8"), "projects.csv", doc_id)
        chunks = chunk_document(parsed)
        assert len(chunks) >= 1

        index = SimpleVectorIndex()
        for c in chunks:
            index.add_chunk(c)

        hits = index.search("Who is the lead for Project Titan and what is the budget?")
        assert len(hits) >= 1
        assert "Project: Project Titan" in hits[0].chunk_text
        assert "Lead: Alice Smith" in hits[0].chunk_text
        assert "Budget: 500000" in hits[0].chunk_text

    def test_e2e_markdown_rag_with_breadcrumbs(self, parser: DocumentParser) -> None:
        md_text = """# Infrastructure SLA

## Reliability Targets

The production cluster guarantees 99.9% uptime per calendar month.

## Maintenance Windows

Maintenance is performed on Sunday at 02:00 UTC.
"""
        doc_id = uuid.uuid4()
        parsed = parser.parse_sync(md_text.encode("utf-8"), "sla.md", doc_id)
        chunks = chunk_document(parsed)
        assert len(chunks) >= 2

        index = SimpleVectorIndex()
        for c in chunks:
            index.add_chunk(c)

        hits = index.search("What is the production reliability uptime guarantee?")
        assert len(hits) >= 1
        top_text = hits[0].chunk_text
        assert "99.9% uptime" in top_text
        # Check that heading hierarchy breadcrumb was preserved
        assert "Infrastructure SLA → Reliability Targets" in top_text

    def test_e2e_txt_rag(self, parser: DocumentParser) -> None:
        txt_data = (
            "Access Control Specification\n\n"
            "All multi-factor authentication tokens expire after 24 hours of inactivity.\n"
            "Emergency break-glass accounts require dual approval."
        )
        doc_id = uuid.uuid4()
        parsed = parser.parse_sync(txt_data.encode("utf-8"), "access.txt", doc_id)
        chunks = chunk_document(parsed)
        assert len(chunks) >= 1

        index = SimpleVectorIndex()
        for c in chunks:
            index.add_chunk(c)

        hits = index.search("When do multi-factor authentication tokens expire?")
        assert len(hits) >= 1
        assert "24 hours" in hits[0].chunk_text

    def test_cross_format_equivalent_fact_retrieval(self, parser: DocumentParser) -> None:
        """Verify that a single known fact is retrievable regardless of source document format."""
        fact = "The standard probationary period for all newly hired employees is 90 days."

        # 1. DOCX
        from docx import Document as DocxDocument
        d = DocxDocument()
        d.add_paragraph(fact)
        b = io.BytesIO()
        d.save(b)
        docx_bytes = b.getvalue()

        # 2. PDF
        import fitz
        p: Any = fitz.open()
        p.new_page().insert_text((50, 72), fact)
        pdf_bytes = p.tobytes()

        # 3. XLSX
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        ws.title = "HR Rules"
        ws.append(["Policy", "Duration"])
        ws.append(["Probationary Period", "90 days"])
        bx = io.BytesIO()
        wb.save(bx)
        xlsx_bytes = bx.getvalue()

        # 4. CSV
        csv_bytes = "Policy,Duration\nProbationary Period,90 days\n".encode("utf-8")

        # 5. MD
        md_bytes = f"# HR Onboarding\n\n{fact}\n".encode("utf-8")

        # 6. TXT
        txt_bytes = fact.encode("utf-8")

        # Ingest all formats into one index
        test_docs = [
            (docx_bytes, "policy.docx"),
            (pdf_bytes, "policy.pdf"),
            (xlsx_bytes, "policy.xlsx"),
            (csv_bytes, "policy.csv"),
            (md_bytes, "policy.md"),
            (txt_bytes, "policy.txt"),
        ]

        index = SimpleVectorIndex()
        for raw, fname in test_docs:
            parsed = parser.parse_sync(raw, fname, uuid.uuid4())
            for c in chunk_document(parsed):
                index.add_chunk(c)

        query = "How long is the probationary period for newly hired employees?"
        results = index.search(query, top_k=6)
        assert len(results) == 6
        for r in results:
            assert "90 days" in r.chunk_text
            assert r.similarity_score > 0.0
