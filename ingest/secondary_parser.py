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


def _parse_pdf(path: Path) -> ParsedDoc:
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    text = "\n\n".join(pages).strip()
    title = _guess_title(text, path.stem)
    year = _guess_year(text)
    return ParsedDoc(
        text=text,
        title=title,
        authors=[],
        publication_year=year,
        parse_method="pymupdf",
        page_count=len(pages),
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
