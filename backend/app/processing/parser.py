"""Extract raw text from supported document formats."""
from __future__ import annotations

import io
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import PdfReadError

_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt", ".md", ".markdown"})


class ParsingError(Exception):
    """Base class for document parsing failures."""


class UnsupportedFormatError(ParsingError):
    """The file extension is not supported by the parser."""


class CorruptedFileError(ParsingError):
    """The file is recognized but cannot be read (corrupt or invalid content)."""


def parse_file(content: bytes, filename: str, mime_type: str | None = None) -> str:
    """Extract raw text from an uploaded file.

    Args:
        content: Raw file bytes.
        filename: Original filename (used to detect format by extension).
        mime_type: Optional MIME type (extension is the primary signal).

    Returns:
        Raw extracted text (not yet cleaned).

    Raises:
        UnsupportedFormatError: Extension is not in the supported set.
        CorruptedFileError: File is recognized but unreadable.
        ParsingError: Other parsing failures.
    """
    extension = Path(filename).suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file extension {extension!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )

    if not content:
        raise CorruptedFileError(f"File {filename!r} is empty and cannot be parsed.")

    try:
        if extension == ".pdf":
            return _parse_pdf(content, filename)
        if extension == ".docx":
            return _parse_docx(content, filename)
        return _parse_plain_text(content, filename)
    except ParsingError:
        raise
    except Exception as exc:
        raise CorruptedFileError(f"Failed to parse {filename!r}: {exc}") from exc


def _parse_pdf(content: bytes, filename: str) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
    except PdfReadError as exc:
        raise CorruptedFileError(f"PDF file {filename!r} is corrupted or invalid: {exc}") from exc

    if len(reader.pages) == 0:
        raise CorruptedFileError(f"PDF file {filename!r} contains no pages.")

    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            parts.append(page_text)

    if not parts:
        raise CorruptedFileError(f"PDF file {filename!r} contains no extractable text.")

    return "\n\n".join(parts)


def _parse_docx(content: bytes, filename: str) -> str:
    try:
        document = DocxDocument(io.BytesIO(content))
    except Exception as exc:
        raise CorruptedFileError(f"DOCX file {filename!r} is corrupted or invalid: {exc}") from exc

    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    if not parts:
        raise CorruptedFileError(f"DOCX file {filename!r} contains no extractable text.")

    return "\n".join(parts)


def _parse_plain_text(content: bytes, filename: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
            if text.strip():
                return text
            raise CorruptedFileError(f"Text file {filename!r} contains no extractable text.")
        except UnicodeDecodeError:
            continue

    raise CorruptedFileError(f"Text file {filename!r} uses an unsupported text encoding.")
