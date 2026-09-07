"""Extract raw text from supported document formats."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import PdfReadError

_SUPPORTED_EXTENSIONS = frozenset({
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".xlsx",
    ".xls",
    ".pptx",
    ".json",
    ".log",
    ".html",
    ".htm",
})


class ParsingError(Exception):
    """Base class for document parsing failures."""


class UnsupportedFormatError(ParsingError):
    """The file extension is not supported by the parser."""


class CorruptedFileError(ParsingError):
    """The file is recognized but cannot be read (corrupt or invalid content)."""


def parse_file(content: bytes, filename: str, mime_type: str | None = None) -> str:
    """Extract raw text from an uploaded file."""
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
        if extension == ".doc":
            return _parse_doc(content, filename)
        if extension in (".xlsx", ".xls"):
            return _parse_spreadsheet(content, filename, extension)
        if extension == ".csv":
            return _parse_csv(content, filename)
        if extension == ".pptx":
            return _parse_pptx(content, filename)
        return _parse_plain_text(content, filename)
    except ParsingError:
        raise
    except Exception as exc:
        raise CorruptedFileError(f"Failed to parse {filename!r}: {exc}") from exc


def _parse_pdf(content: bytes, filename: str) -> str:
    parts: list[str] = []

    # 1. Primary: PyMuPDF (fitz) if installed for rich text & layout extraction
    try:
        import importlib

        fitz = importlib.import_module("fitz")
        pdf_doc = fitz.open(stream=content, filetype="pdf")
        for page in pdf_doc:
            page_text = page.get_text()
            if page_text and page_text.strip():
                parts.append(page_text.strip())
        if parts:
            return "\n\n".join(parts)
    except Exception:
        parts = []

    # 2. Secondary fallback: pypdf
    try:
        reader = PdfReader(io.BytesIO(content))
    except PdfReadError as exc:
        raise CorruptedFileError(f"PDF file {filename!r} is corrupted or invalid: {exc}") from exc

    if len(reader.pages) == 0:
        raise CorruptedFileError(f"PDF file {filename!r} contains no pages.")

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text and page_text.strip():
            parts.append(page_text.strip())

    if not parts:
        raise CorruptedFileError("No extractable text found. This PDF may require OCR.")

    return "\n\n".join(parts)


def _parse_docx(content: bytes, filename: str) -> str:
    try:
        document = DocxDocument(io.BytesIO(content))
    except Exception as exc:
        raise CorruptedFileError(f"DOCX file {filename!r} is corrupted or invalid: {exc}") from exc

    parts: list[str] = []

    # 1. Extract paragraphs
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    # 2. Extract table cell contents
    for table in document.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                # Deduplicate identical adjacent cell text from merged cells
                deduped: list[str] = []
                for cell_text in row_cells:
                    if not deduped or deduped[-1] != cell_text:
                        deduped.append(cell_text)
                parts.append(" | ".join(deduped))

    if not parts:
        raise CorruptedFileError(f"DOCX file {filename!r} contains no extractable text.")

    extracted_text = "\n".join(parts).strip()
    if not extracted_text:
        raise CorruptedFileError(f"DOCX file {filename!r} resulted in 0 extracted characters.")

    return extracted_text


def _parse_doc(content: bytes, filename: str) -> str:
    if content.startswith(b"PK\x03\x04"):
        return _parse_docx(content, filename)
    try:
        from sharepoint2text import read_bytes  # type: ignore[import-untyped]
        docs = list(read_bytes(content, extension=".doc"))
        parts: list[str] = []
        for doc in docs:
            if hasattr(doc, "units") and doc.units:
                for unit in doc.units:
                    t = getattr(unit, "text", "") or ""
                    if t and t.strip():
                        parts.append(t.strip())
            elif hasattr(doc, "full_text") and doc.full_text and doc.full_text.strip():
                parts.append(doc.full_text.strip())
        if not parts:
            raise CorruptedFileError(f"DOC file {filename!r} contains no extractable text.")
        return "\n\n".join(parts)
    except ParsingError:
        raise
    except Exception as exc:
        raise CorruptedFileError(f"DOC file {filename!r} is corrupted or cannot be parsed: {exc}") from exc


def _format_structured_rows(sheet_name: str, rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header_idx = -1
    headers: list[str] = []
    for idx, row in enumerate(rows):
        non_empty = [str(c).strip() for c in row if c is not None and str(c).strip()]
        if non_empty:
            header_idx = idx
            headers = [
                str(c).strip() if c is not None and str(c).strip() else f"Column {col_i + 1}"
                for col_i, c in enumerate(row)
            ]
            break

    if header_idx == -1:
        return ""

    lines: list[str] = [f"--- Sheet: {sheet_name} ---"]
    for row in rows[header_idx + 1:]:
        pairs = []
        for col_i, cell in enumerate(row):
            if cell is not None:
                val_str = str(cell).strip()
                if val_str:
                    if isinstance(cell, float) and cell.is_integer():
                        val_str = str(int(cell))
                    h = headers[col_i] if col_i < len(headers) else f"Column {col_i + 1}"
                    pairs.append(f"{h}: {val_str}")
        if pairs:
            lines.append(" | ".join(pairs))

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _parse_spreadsheet(content: bytes, filename: str, extension: str) -> str:
    parts: list[str] = []
    if extension == ".xlsx":
        try:
            import openpyxl

            workbook = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                rows = [list(r) for r in sheet.iter_rows(values_only=True)]
                sheet_text = _format_structured_rows(sheet_name, rows)
                if sheet_text:
                    parts.append(sheet_text)
        except Exception as exc:
            raise CorruptedFileError(f"XLSX file {filename!r} is corrupted or invalid: {exc}") from exc
    elif extension == ".xls":
        try:
            import xlrd  # type: ignore[import-untyped]

            workbook = xlrd.open_workbook(file_contents=content)
            for sheet_name in workbook.sheet_names():
                sheet = workbook.sheet_by_name(sheet_name)
                rows = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
                sheet_text = _format_structured_rows(sheet_name, rows)
                if sheet_text:
                    parts.append(sheet_text)
        except Exception as exc:
            raise CorruptedFileError(f"XLS file {filename!r} is corrupted or invalid: {exc}") from exc

    if not parts:
        raise CorruptedFileError(f"Spreadsheet file {filename!r} contains no extractable data.")
    return "\n\n".join(parts)


def _parse_csv(content: bytes, filename: str) -> str:
    import csv

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise CorruptedFileError(f"CSV file {filename!r} uses an unsupported text encoding.")

    if not text.strip():
        raise CorruptedFileError(f"CSV file {filename!r} is empty.")

    delimiter = ","
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except Exception:
        first_line = text.split("\n", 1)[0]
        counts = {d: first_line.count(d) for d in (",", ";", "\t", "|")}
        best_delim = max(counts, key=lambda d: counts[d])
        if counts[best_delim] > 0:
            delimiter = best_delim

    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = [list(r) for r in reader]
    except Exception as exc:
        raise CorruptedFileError(f"CSV file {filename!r} is corrupted or invalid: {exc}") from exc

    base_name = Path(filename).stem
    formatted = _format_structured_rows(base_name, rows)
    if not formatted:
        raise CorruptedFileError(f"CSV file {filename!r} contains no extractable data.")
    return formatted


def _parse_pptx(content: bytes, filename: str) -> str:
    try:
        from pptx import Presentation

        prs = Presentation(io.BytesIO(content))
        parts: list[str] = []
        for slide_idx, slide in enumerate(prs.slides, 1):
            parts.append(f"--- Slide {slide_idx} ---")
            for shape in slide.shapes:
                text_val = getattr(shape, "text", None)
                if text_val and text_val.strip():
                    parts.append(text_val.strip())
        if not parts:
            raise CorruptedFileError(f"PPTX file {filename!r} contains no extractable text.")
        return "\n".join(parts)
    except Exception as exc:
        raise CorruptedFileError(f"PPTX file {filename!r} is corrupted or invalid: {exc}") from exc


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
