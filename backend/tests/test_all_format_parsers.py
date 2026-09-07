"""Unit test suite for all 8 document formats and parser edge cases.

Validates parsing and structure preservation for:
  - .docx
  - .doc (Word 97-2003 OLE binary)
  - .pdf
  - .xlsx
  - .xls (Excel 97-2003 binary)
  - .csv
  - .md
  - .txt
Plus negative and edge cases:
  - Scanned/empty PDF OCR requirement notice
  - Empty files
  - Corrupted files
  - Unsupported extensions
  - Missing CSV values
"""
from __future__ import annotations

import io
import struct
import uuid
from typing import Any
import pytest

from app.processing.parser import CorruptedFileError, UnsupportedFormatError, parse_file
from app.services.metadata import BlockType
from app.services.parser import DocumentParser


def _make_minimal_ole_doc(text: str) -> bytes:
    """Construct a valid OLE Compound Document containing a WordDocument stream."""
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


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser()


class TestAllFormatParsers:
    """Comprehensive test matrix for all 8 supported document formats."""

    def test_docx_parsing(self, parser: DocumentParser) -> None:
        """Validate DOCX headings, paragraphs, lists, tables, and unicode."""
        from docx import Document as DocxDocument

        doc = DocxDocument()
        doc.add_heading("Company Leave Policy", level=1)
        doc.add_paragraph("Employees are eligible for paid time off under the 2026 regulations.")
        doc.add_paragraph("Annual leave is 18 days.", style="List Bullet")
        doc.add_paragraph("Sick leave is 12 days.", style="List Bullet")

        table = doc.add_table(rows=2, cols=3)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Tier"
        hdr_cells[1].text = "Department"
        hdr_cells[2].text = "Days"

        row_cells = table.rows[1].cells
        row_cells[0].text = "Standard"
        row_cells[1].text = "Engineering"
        row_cells[2].text = "18"

        buf = io.BytesIO()
        doc.save(buf)
        docx_bytes = buf.getvalue()

        parsed = parser.parse_sync(docx_bytes, "leave_policy.docx", uuid.uuid4())
        assert len(parsed.blocks) >= 3

        heading_blocks = [b for b in parsed.blocks if b.block_type == BlockType.HEADING]
        assert len(heading_blocks) >= 1
        assert "Company Leave Policy" in heading_blocks[0].text

        table_blocks = [b for b in parsed.blocks if b.block_type == BlockType.TABLE]
        assert len(table_blocks) == 1
        assert "Tier" in table_blocks[0].text
        assert "Engineering" in table_blocks[0].text

        raw_text = parse_file(docx_bytes, "leave_policy.docx")
        assert "Company Leave Policy" in raw_text
        assert "18 days" in raw_text

    def test_doc_binary_ole_parsing(self, parser: DocumentParser) -> None:
        """Validate Word 97-2003 OLE binary (.doc) extraction."""
        fact_text = "Travel policy: The standard international per diem rate is 120 USD."
        doc_bytes = _make_minimal_ole_doc(fact_text)

        parsed = parser.parse_sync(doc_bytes, "travel_policy.doc", uuid.uuid4())
        assert len(parsed.blocks) >= 1
        combined_text = "\n".join(b.text for b in parsed.blocks)
        assert "120 USD" in combined_text

        raw_text = parse_file(doc_bytes, "travel_policy.doc")
        assert "120 USD" in raw_text

    def test_doc_renamed_docx_fallback(self, parser: DocumentParser) -> None:
        """Validate that a modern docx file uploaded with .doc extension still parses."""
        from docx import Document as DocxDocument

        doc = DocxDocument()
        doc.add_heading("Safety Guidelines", level=1)
        doc.add_paragraph("Evacuation assembly point is Zone 4.")
        buf = io.BytesIO()
        doc.save(buf)
        docx_bytes = buf.getvalue()

        parsed = parser.parse_sync(docx_bytes, "safety.doc", uuid.uuid4())
        assert len(parsed.blocks) >= 1
        assert "Zone 4" in parsed.blocks[-1].text

    def test_pdf_parsing(self, parser: DocumentParser) -> None:
        """Validate multi-page PDF with headings, lists, and unicode."""
        import fitz

        pdf_doc: Any = fitz.open()
        page1 = pdf_doc.new_page()
        page1.insert_text((50, 72), "Medical Benefits Overview", fontsize=18)
        page1.insert_text((50, 120), "The comprehensive plan includes dental coverage of $1500 annually.", fontsize=11)

        page2 = pdf_doc.new_page()
        page2.insert_text((50, 72), "Vision Care Coverage", fontsize=18)
        page2.insert_text((50, 120), "Frames are covered up to $250 every 24 months.", fontsize=11)

        pdf_bytes = pdf_doc.tobytes()

        parsed = parser.parse_sync(pdf_bytes, "benefits.pdf", uuid.uuid4())
        assert parsed.page_count == 2
        combined_text = "\n".join(b.text for b in parsed.blocks)
        assert "Medical Benefits Overview" in combined_text
        assert "1500" in combined_text
        assert "250" in combined_text

        raw_text = parse_file(pdf_bytes, "benefits.pdf")
        assert "1500" in raw_text

    def test_pdf_scanned_ocr_required_notice(self, parser: DocumentParser) -> None:
        """Validate that an empty/scanned PDF raises the explicit OCR notice."""
        import fitz

        pdf_doc: Any = fitz.open()
        pdf_doc.new_page()  # Page without text (scanned image emulation)
        pdf_bytes = pdf_doc.tobytes()

        with pytest.raises(CorruptedFileError) as exc_info:
            parser.parse_sync(pdf_bytes, "scanned_doc.pdf", uuid.uuid4())
        assert "No extractable text found. This PDF may require OCR." in str(exc_info.value)

        with pytest.raises(CorruptedFileError) as exc_info_raw:
            parse_file(pdf_bytes, "scanned_doc.pdf")
        assert "No extractable text found. This PDF may require OCR." in str(exc_info_raw.value)

    def test_xlsx_structured_parsing(self, parser: DocumentParser) -> None:
        """Validate .xlsx worksheet parsing, preserving header key-value relationships."""
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        ws.title = "Payroll Data"
        ws.append(["Employee", "Department", "Leave Balance", "Years of Service"])
        ws.append(["Arun", "IT", 12, 4])
        ws.append(["Priya", "HR", 15, 6])
        ws.append(["Vikram", "Finance", 20, 8])

        buf = io.BytesIO()
        wb.save(buf)
        xlsx_bytes = buf.getvalue()

        parsed = parser.parse_sync(xlsx_bytes, "payroll.xlsx", uuid.uuid4())
        assert len(parsed.blocks) == 1
        block = parsed.blocks[0]
        assert block.block_type == BlockType.TABLE
        assert block.metadata["sheet"] == "Payroll Data"
        assert block.metadata["row_start"] == 2
        assert block.metadata["row_end"] == 4

        lines = block.text.splitlines()
        assert lines[0].startswith("Sheet: Payroll Data")
        assert "Employee: Arun | Department: IT | Leave Balance: 12 | Years of Service: 4" in lines[1]
        assert "Employee: Priya | Department: HR | Leave Balance: 15 | Years of Service: 6" in lines[2]

        raw_text = parse_file(xlsx_bytes, "payroll.xlsx")
        assert "Employee: Arun | Department: IT | Leave Balance: 12" in raw_text

    def test_xls_structured_parsing(self, parser: DocumentParser) -> None:
        """Validate legacy .xls worksheet parsing with xlrd and header preservation."""
        import xlwt  # type: ignore[import-untyped]

        wb = xlwt.Workbook()
        ws = wb.add_sheet("Inventory")
        ws.write(0, 0, "Item Code")
        ws.write(0, 1, "Description")
        ws.write(0, 2, "Stock Count")
        ws.write(1, 0, "SKU-9920")
        ws.write(1, 1, "Mechanical Widget")
        ws.write(1, 2, 450)

        buf = io.BytesIO()
        wb.save(buf)
        xls_bytes = buf.getvalue()

        parsed = parser.parse_sync(xls_bytes, "inventory.xls", uuid.uuid4())
        assert len(parsed.blocks) == 1
        block = parsed.blocks[0]
        assert block.block_type == BlockType.TABLE
        assert block.metadata["sheet"] == "Inventory"

        assert "Item Code: SKU-9920 | Description: Mechanical Widget | Stock Count: 450" in block.text

        raw_text = parse_file(xls_bytes, "inventory.xls")
        assert "Item Code: SKU-9920 | Description: Mechanical Widget | Stock Count: 450" in raw_text

    def test_csv_structured_parsing_delimiters(self, parser: DocumentParser) -> None:
        """Validate .csv parsing across comma, semicolon, and tab delimiters."""
        csv_comma = "Employee,Department,Leave Balance\nArun,IT,12\nPriya,HR,15\n"
        parsed_comma = parser.parse_sync(csv_comma.encode("utf-8"), "data.csv", uuid.uuid4())
        assert "Employee: Arun | Department: IT | Leave Balance: 12" in parsed_comma.blocks[0].text

        csv_semi = "Product;Category;Price\nLaptop;Electronics;1200\nMouse;Accessories;25\n"
        parsed_semi = parser.parse_sync(csv_semi.encode("utf-8"), "products.csv", uuid.uuid4())
        assert "Product: Laptop | Category: Electronics | Price: 1200" in parsed_semi.blocks[0].text

        csv_tab = "City\tCountry\tPopulation\nBerlin\tGermany\t3700000\nParis\tFrance\t2161000\n"
        parsed_tab = parser.parse_sync(csv_tab.encode("utf-8"), "cities.csv", uuid.uuid4())
        assert "City: Berlin | Country: Germany | Population: 3700000" in parsed_tab.blocks[0].text

    def test_markdown_parsing_hierarchy_and_tables(self, parser: DocumentParser) -> None:
        """Validate markdown structure: heading hierarchy, code fence language, and tables."""
        md_content = """# Infrastructure Architecture

## Core Services

The backend service runs on port 8000 with asynchronous database pooling.

### Configuration Snippet

```yaml
server:
  port: 8000
  workers: 4
```

## Team Allocations

| Role | Engineer | Allocation |
| --- | --- | --- |
| Tech Lead | Aravind | 100% |
| DevOps | David | 50% |
"""
        parsed = parser.parse_sync(md_content.encode("utf-8"), "architecture.md", uuid.uuid4())

        headings = [b for b in parsed.blocks if b.block_type in (BlockType.HEADING, BlockType.SUBHEADING)]
        assert len(headings) >= 3
        assert headings[0].text == "Infrastructure Architecture"
        assert headings[1].text == "Core Services"
        assert headings[2].text == "Configuration Snippet"

        code_blocks = [b for b in parsed.blocks if b.block_type == BlockType.CODE]
        assert len(code_blocks) >= 1
        assert "port: 8000" in code_blocks[0].text
        assert code_blocks[0].metadata.get("language") == "yaml"

        table_blocks = [b for b in parsed.blocks if b.block_type == BlockType.TABLE]
        assert len(table_blocks) == 1
        assert "Role: Tech Lead | Engineer: Aravind | Allocation: 100%" in table_blocks[0].text

    def test_txt_parsing_encodings(self, parser: DocumentParser) -> None:
        """Validate plain text parsing with Unicode and UTF-8-SIG."""
        text_content = (
            "System Security Policy\n\n"
            "All passwords must be rotated every 90 days.\n"
            "Special characters such as ©, ®, and € are fully supported."
        )

        utf8_sig = text_content.encode("utf-8-sig")
        parsed = parser.parse_sync(utf8_sig, "security.txt", uuid.uuid4())
        assert len(parsed.blocks) >= 2
        combined = "\n".join(b.text for b in parsed.blocks)
        assert "90 days" in combined
        assert "€" in combined


class TestNegativeParserCases:
    """Validate graceful error handling across edge cases and malformed inputs."""

    def test_empty_file_rejection(self, parser: DocumentParser) -> None:
        with pytest.raises(CorruptedFileError):
            parser.parse_sync(b"", "empty.txt", uuid.uuid4())

        with pytest.raises(CorruptedFileError):
            parse_file(b"", "empty.txt")

    def test_corrupted_binary_rejection(self, parser: DocumentParser) -> None:
        garbage = b"\x00\x01\x02\x03\x04\x05GARBAGE_CONTENT"
        with pytest.raises(CorruptedFileError):
            parser.parse_sync(garbage, "corrupt.docx", uuid.uuid4())

        with pytest.raises(CorruptedFileError):
            parser.parse_sync(garbage, "corrupt.xlsx", uuid.uuid4())

        with pytest.raises(CorruptedFileError):
            parser.parse_sync(garbage, "corrupt.xls", uuid.uuid4())

    def test_unsupported_extension_rejection(self, parser: DocumentParser) -> None:
        with pytest.raises(UnsupportedFormatError):
            parser.parse_sync(b"binary", "executable.exe", uuid.uuid4())

        with pytest.raises(UnsupportedFormatError):
            parse_file(b"archive", "archive.tar.gz")

    def test_empty_excel_sheet(self, parser: DocumentParser) -> None:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        buf = io.BytesIO()
        wb.save(buf)

        with pytest.raises(CorruptedFileError):
            parser.parse_sync(buf.getvalue(), "blank.xlsx", uuid.uuid4())

    def test_csv_with_missing_cells(self, parser: DocumentParser) -> None:
        """Verify that sparse CSV rows with missing columns do not crash."""
        csv_sparse = "Name,Age,Role\nArun,,Engineer\n,30,\n"
        parsed = parser.parse_sync(csv_sparse.encode("utf-8"), "sparse.csv", uuid.uuid4())
        assert len(parsed.blocks) == 1
        assert "Name: Arun | Role: Engineer" in parsed.blocks[0].text
