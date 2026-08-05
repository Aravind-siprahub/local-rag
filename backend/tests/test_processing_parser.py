"""Unit tests for `app.processing.parser`."""
import io

import pytest
from docx import Document as DocxDocument

from app.processing.parser import (
    CorruptedFileError,
    UnsupportedFormatError,
    parse_file,
)

# Minimal valid PDF with extractable text "Hello PDF" (Helvetica, no external deps).
_MINIMAL_PDF = b"""%PDF-1.1
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 24 Tf 20 100 Td (Hello PDF) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
0000000324 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
398
%%EOF"""


def _make_docx(paragraphs: list[str]) -> bytes:
  document = DocxDocument()
  for paragraph in paragraphs:
    document.add_paragraph(paragraph)
  buffer = io.BytesIO()
  document.save(buffer)
  return buffer.getvalue()


class TestParseFile:
  def test_parses_txt(self) -> None:
    result = parse_file(b"Hello world\nSecond line", "notes.txt")
    assert "Hello world" in result
    assert "Second line" in result

  def test_parses_markdown(self) -> None:
    content = b"# Title\n\nSome **markdown** content."
    result = parse_file(content, "readme.md")
    assert "Title" in result
    assert "markdown" in result

  def test_parses_markdown_extension(self) -> None:
    result = parse_file(b"markdown body", "notes.markdown")
    assert result == "markdown body"

  def test_parses_pdf(self) -> None:
    result = parse_file(_MINIMAL_PDF, "report.pdf", "application/pdf")
    assert "Hello PDF" in result

  def test_parses_docx(self) -> None:
    docx_bytes = _make_docx(["First paragraph", "Second paragraph"])
    result = parse_file(docx_bytes, "report.docx")
    assert "First paragraph" in result
    assert "Second paragraph" in result

  def test_rejects_unsupported_extension(self) -> None:
    with pytest.raises(UnsupportedFormatError, match="Unsupported file extension"):
      parse_file(b"data", "virus.exe")

  def test_rejects_empty_file(self) -> None:
    with pytest.raises(CorruptedFileError, match="empty"):
      parse_file(b"", "empty.txt")

  def test_rejects_corrupted_pdf(self) -> None:
    with pytest.raises(CorruptedFileError, match="corrupted|invalid|Failed"):
      parse_file(b"not a pdf", "broken.pdf")

  def test_rejects_blank_txt(self) -> None:
    with pytest.raises(CorruptedFileError, match="no extractable text"):
      parse_file(b"   \n  \n", "blank.txt")

  def test_rejects_blank_docx(self) -> None:
    docx_bytes = _make_docx([])
    with pytest.raises(CorruptedFileError, match="no extractable text"):
      parse_file(docx_bytes, "empty.docx")
