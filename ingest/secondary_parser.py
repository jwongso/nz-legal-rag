"""
Parse secondary source files (PDF, DOCX, TXT, MD) into plain text.

Returns a ParsedDoc with raw text and best-effort metadata (title, authors, year).
Heavy metadata extraction (LLM-assisted) is out of scope here - this is the
cheap heuristic pass that gets text ready for chunking.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDoc:
    text: str
    title: str
    authors: list[str]
    publication_year: int | None
    parse_method: str
    page_count: int | None = None


_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def _guess_year(text: str) -> int | None:
    m = _YEAR_RE.search(text[:2000])
    return int(m.group()) if m else None


def _guess_title(text: str, stem: str) -> str:
    """Use first non-empty line of document, fall back to filename stem."""
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 10:
            return line[:200]
    return stem


_OCR_CHARS_PER_PAGE_THRESHOLD = 200  # below this avg triggers OCR fallback


def _ocr_pdf(path: Path) -> tuple[list[str], int]:
    """Render each page to image and OCR it. Returns (page_texts, page_count)."""
    import fitz
    import pytesseract
    from PIL import Image
    import io

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages.append(pytesseract.image_to_string(img, lang="eng"))
    doc.close()
    return pages, len(pages)


def _parse_pdf(path: Path) -> ParsedDoc:
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    page_count = len(pages)
    doc.close()

    text = "\n\n".join(pages).strip()
    avg_chars = len(text) / max(page_count, 1)
    parse_method = "pymupdf"

    if avg_chars < _OCR_CHARS_PER_PAGE_THRESHOLD:
        ocr_pages, page_count = _ocr_pdf(path)
        text = "\n\n".join(ocr_pages).strip()
        parse_method = "pymupdf+ocr"

    title = _guess_title(text, path.stem)
    year = _guess_year(text)
    return ParsedDoc(
        text=text,
        title=title,
        authors=[],
        publication_year=year,
        parse_method=parse_method,
        page_count=page_count,
    )


def _parse_docx(path: Path) -> ParsedDoc:
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    title = _guess_title(text, path.stem)
    year = _guess_year(text)
    return ParsedDoc(
        text=text,
        title=title,
        authors=[],
        publication_year=year,
        parse_method="python-docx",
    )


def _parse_text(path: Path) -> ParsedDoc:
    text = path.read_text(errors="replace").strip()
    title = _guess_title(text, path.stem)
    year = _guess_year(text)
    return ParsedDoc(
        text=text,
        title=title,
        authors=[],
        publication_year=year,
        parse_method="plaintext",
    )


def parse(path: Path) -> ParsedDoc:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(path)
    elif suffix in (".txt", ".md", ".rst", ".html"):
        return _parse_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
