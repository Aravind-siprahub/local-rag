"""Centralized RAG citation generation, formatting, and anti-hallucination validation."""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)


def build_citation_label(
    document_title: str | None,
    page_number: int | None = None,
    section_title: str | None = None,
    source_location: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Build a standardized, accurate citation label without fabricating metadata.

    Formats:
        - PDF: [Document.pdf, p. 5]
        - DOCX: [Document.docx, p. 8] or [Document.docx, Section: Leave Policy]
        - Markdown: [Document.md, Production Deployment]
        - Excel: [Document.xlsx, Sheet: Q2]
        - Fallback: [Document.ext]
    """
    doc_name = (document_title or "Document").strip()
    # Strip any brackets user might have had in the document name
    doc_name = doc_name.strip("[]")
    ext = doc_name.split(".")[-1].lower() if "." in doc_name else ""

    # 1. If explicit source_location is provided (e.g. from chunk metadata)
    if source_location and source_location.strip():
        loc = source_location.strip()
        for sep in ("→", "->", ">"):
            if sep in loc:
                loc = loc.split(sep)[-1].strip()
        return f"[{doc_name}, {loc}]"

    # 2. Check metadata dictionary if present
    meta = metadata or {}
    sheet = meta.get("sheet")
    r_start = meta.get("row_start")
    r_end = meta.get("row_end")

    if ext in ("xlsx", "xls", "csv") or sheet:
        if sheet:
            if r_start is not None and r_end is not None:
                return f"[{doc_name}, Sheet: {sheet}, Rows {r_start}-{r_end}]"
            return f"[{doc_name}, Sheet: {sheet}]"

    # 3. PDF format
    if ext == "pdf":
        if page_number is not None and page_number > 0:
            return f"[{doc_name}, p. {page_number}]"
        return f"[{doc_name}]"

    # 4. DOCX / DOC format
    if ext in ("docx", "doc"):
        if page_number is not None and page_number > 0:
            return f"[{doc_name}, p. {page_number}]"
        if section_title and section_title.strip():
            clean_sec = section_title.strip()
            for sep in ("→", "->", ">"):
                if sep in clean_sec:
                    clean_sec = clean_sec.split(sep)[-1].strip()
            prefix = "" if clean_sec.lower().startswith("section:") else "Section: "
            return f"[{doc_name}, {prefix}{clean_sec}]"
        return f"[{doc_name}]"

    # 5. Markdown format
    if ext in ("md", "markdown"):
        if section_title and section_title.strip():
            clean_sec = section_title.strip()
            for sep in ("→", "->", ">"):
                if sep in clean_sec:
                    clean_sec = clean_sec.split(sep)[-1].strip()
            return f"[{doc_name}, {clean_sec}]"
        return f"[{doc_name}]"

    # 6. Generic / TXT fallback
    if page_number is not None and page_number > 0:
        return f"[{doc_name}, p. {page_number}]"
    if section_title and section_title.strip():
        clean_sec = section_title.strip()
        for sep in ("→", "->", ">"):
            if sep in clean_sec:
                clean_sec = clean_sec.split(sep)[-1].strip()
        return f"[{doc_name}, {clean_sec}]"
    if section_title and section_title.strip():
        return f"[{doc_name}, {section_title.strip()}]"

    return f"[{doc_name}]"


# Regex to detect inline citation tags such as:
# [Document.pdf, p. 12] or [Handbook.docx, Section: Leave] or [Doc.md, Intro] or [C1]
_CITATION_PATTERN = re.compile(
    r"\[(?!(?:Truncated|Prior Conversation Summary|Chunk \d+|http|https)\b)([^\[\]\n]+)\]"
)


def extract_inline_citations(text: str) -> list[str]:
    """Extract all inline bracketed citation candidates from text."""
    if not text:
        return []
    return _CITATION_PATTERN.findall(text)


def validate_and_sanitize_citations(
    answer: str,
    retrieved_chunks: list[Any],
    similarity_threshold: float = 0.0,
) -> tuple[str, list[dict[str, Any]]]:
    """Verify generated citations against actual retrieved chunks.

    Removes or corrects hallucinated citations (e.g. non-existent documents,
    unretrieved pages/sections). Ensures every citation in the answer is grounded
    in actual retrieval metadata.

    Returns:
        tuple[str, list[dict]]: (sanitized_answer, verified_citations_list)
    """
    if not answer or not answer.strip():
        return answer, []

    # Map retrieved chunks into searchable lookup records
    valid_records: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        score = float(getattr(chunk, "similarity_score", 1.0))
        if score < similarity_threshold:
            continue

        cid = str(getattr(chunk, "chunk_id", "") or "")
        if cid and cid in seen_chunk_ids:
            continue
        if cid:
            seen_chunk_ids.add(cid)

        title = getattr(chunk, "document_title", None) or "Document"
        page = getattr(chunk, "page_number", None)
        section = getattr(chunk, "section_title", None)
        meta = getattr(chunk, "metadata_", None) or getattr(chunk, "metadata", None) or {}
        source_loc = getattr(chunk, "source_location", None) or meta.get("source_location")

        label = build_citation_label(
            document_title=title,
            page_number=page,
            section_title=section,
            source_location=source_loc,
            metadata=meta,
        )

        doc_id = getattr(chunk, "document_id", None)
        if isinstance(doc_id, uuid.UUID):
            pass
        elif doc_id:
            try:
                doc_id = uuid.UUID(str(doc_id))
            except Exception:
                doc_id = uuid.uuid4()
        else:
            doc_id = uuid.uuid4()

        valid_records.append({
            "chunk_id": uuid.UUID(cid) if cid and len(cid) == 36 else (uuid.uuid5(uuid.NAMESPACE_DNS, cid) if cid else uuid.uuid4()),
            "chunk_text": getattr(chunk, "chunk_text", "") or "",
            "document_id": doc_id,
            "document_version_id": getattr(chunk, "document_version_id", None),
            "document_title": title,
            "document_name": title,
            "file_name": title,
            "section_title": section,
            "section": section,
            "page_number": page,
            "source_location": source_loc or (label.strip("[]").split(",", 1)[-1].strip() if "," in label else None),
            "similarity_score": score,
            "relevance_score": score,
            "rank": getattr(chunk, "rank", idx),
            "citation_id": f"C{idx}",
            "citation_label": label,
            "raw_chunk": chunk,
        })

    if not valid_records:
        # No valid retrieved chunks — strip any fabricated citation brackets
        sanitized = _CITATION_PATTERN.sub("", answer)
        sanitized = re.sub(r" +", " ", sanitized).replace(" .", ".").strip()
        return sanitized, []

    # Check each bracket match in answer
    verified_citations: list[dict[str, Any]] = []
    used_chunk_indices: set[int] = set()

    def _replace_or_validate(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        full_tag = f"[{inner}]"

        # Check if it's already a C1, C2... citation ID
        for idx, rec in enumerate(valid_records, start=1):
            if inner.upper() == f"C{idx}":
                used_chunk_indices.add(idx - 1)
                return rec["citation_label"]

        # Match by document filename and location
        parts = [p.strip() for p in inner.split(",", 1)]
        cited_doc = parts[0]
        cited_loc = parts[1] if len(parts) > 1 else ""

        # Find best matching chunk
        matched_chunk_idx: int | None = None
        for i, rec in enumerate(valid_records):
            rec_title = (rec["document_title"] or "").lower()
            rec_file = (rec["file_name"] or "").lower()
            cited_lower = cited_doc.lower()

            if (cited_lower == rec_title or cited_lower == rec_file
                    or cited_lower in rec_title or rec_title in cited_lower):
                # Verify page or location if specified
                if cited_loc:
                    # Check if page matches
                    page_m = re.search(r"\bp(?:age|\.)?\s*(\d+)\b", cited_loc, re.IGNORECASE)
                    if page_m:
                        cited_page = int(page_m.group(1))
                        if rec["page_number"] == cited_page:
                            matched_chunk_idx = i
                            break
                        # Page mismatch: if this doc only has 1 page in retrieval, we can correct it
                        # otherwise mark as unverified
                        continue

                    # Check if sheet matches
                    sheet_m = re.search(r"\bsheet:\s*([^\n\r,\)]+)", cited_loc, re.IGNORECASE)
                    if sheet_m and rec["source_location"]:
                        if sheet_m.group(1).lower() in rec["source_location"].lower():
                            matched_chunk_idx = i
                            break
                        continue

                    # Check section
                    if rec["section_title"] and (cited_loc.lower() in rec["section_title"].lower() or rec["section_title"].lower() in cited_loc.lower()):
                        matched_chunk_idx = i
                        break

                # If no location specified in citation or document matched cleanly
                matched_chunk_idx = i
                break

        if matched_chunk_idx is not None:
            rec = valid_records[matched_chunk_idx]
            used_chunk_indices.add(matched_chunk_idx)
            # Return standardized canonical citation label
            return rec["citation_label"]

        # Citation could not be verified against retrieved chunks -> hallucination! Strip it.
        logger.warning("Stripped hallucinated citation: %r", full_tag)
        return ""

    sanitized_answer = _CITATION_PATTERN.sub(_replace_or_validate, answer)
    # Clean up any leftover punctuation or whitespace from stripped citations
    sanitized_answer = re.sub(r" +", " ", sanitized_answer)
    sanitized_answer = re.sub(r"\s+([.,;:!?])", r"\1", sanitized_answer)

    # If the LLM didn't generate citations inline, but we have retrieved context that answered the query,
    # attach the citation for the top retrieved chunk so factual grounding is preserved
    if not used_chunk_indices and valid_records:
        top_rec = valid_records[0]
        used_chunk_indices.add(0)
        # Append inline citation to end of answer if appropriate and not an unanswerable message
        unanswerable_triggers = [
            "couldn't find enough information",
            "could not find enough information",
            "no document excerpts were available",
        ]
        is_unanswerable = any(trig in sanitized_answer.lower() for trig in unanswerable_triggers)
        if not is_unanswerable and top_rec["similarity_score"] >= 0.35:
            stripped = sanitized_answer.rstrip()
            last_line = stripped.split("\n")[-1].strip()
            if last_line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "#")):
                sanitized_answer = f"{stripped}\n\n{top_rec['citation_label']}"
            else:
                sanitized_answer = f"{stripped} {top_rec['citation_label']}"

    # Build the citations list for the API response in order of appearance / relevance
    verified_citations = [valid_records[i] for i in sorted(used_chunk_indices)]
    if not verified_citations and valid_records:
        verified_citations = valid_records[:3]

    return sanitized_answer, verified_citations
