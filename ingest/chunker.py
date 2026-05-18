"""
Section-aware chunker for NZ legal decisions.

Legal documents have natural break points: numbered paragraphs, headings, legislation
references. Splitting on these boundaries produces better retrieval than naive sliding
windows because each chunk remains semantically coherent.
"""

import re
from dataclasses import dataclass

import config
from ingest.scraper import CaseDocument


@dataclass
class Chunk:
    chunk_id: str
    case_id: str
    court: str
    court_name: str
    year: int
    title: str
    date: str
    parties: list[str]
    url: str
    text: str
    section_heading: str
    chunk_index: int
    citations: list[str]


_SECTION_PATTERNS = [
    # Numbered headings: "1.", "1.1", "[1]"
    re.compile(r"^(\[?\d+\.?\d*\.?\]?)\s+[A-Z]"),
    # ALL CAPS headings: BACKGROUND, FACTS, DECISION, LAW
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),
    # Mixed case section headings
    re.compile(r"^(Background|Facts|Issues?|Law|Legislation|Analysis|Decision|Orders?|Summary|Introduction)\b", re.I),
]


def _is_section_break(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _SECTION_PATTERNS)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Return list of (heading, body) pairs.
    Heading is empty string for the preamble before the first section.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        if _is_section_break(line):
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body or current_heading:
        sections.append((current_heading, current_body))

    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _split_by_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Sliding window split on word boundaries when a section is too long."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + max_words]
        chunks.append(" ".join(chunk_words))
        i += max_words - overlap_words
    return chunks


def chunk_case(doc: CaseDocument) -> list[Chunk]:
    sections = _split_into_sections(doc.text)
    chunks: list[Chunk] = []
    idx = 0

    for heading, body in sections:
        if _word_count(body) <= config.CHUNK_SIZE:
            text = f"{heading}\n\n{body}".strip() if heading else body
            if _word_count(text) < config.CHUNK_MIN_WORDS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.case_id}#{idx}",
                    case_id=doc.case_id,
                    court=doc.court,
                    court_name=doc.court_name,
                    year=doc.year,
                    title=doc.title,
                    date=doc.date,
                    parties=doc.parties,
                    url=doc.url,
                    text=text,
                    section_heading=heading,
                    chunk_index=idx,
                    citations=doc.citations,
                )
            )
            idx += 1
        else:
            # Section too long: sliding window within the section
            sub_chunks = _split_by_words(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for sub in sub_chunks:
                if _word_count(sub) < config.CHUNK_MIN_WORDS:
                    continue
                prefix = f"{heading}\n\n" if heading else ""
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.case_id}#{idx}",
                        case_id=doc.case_id,
                        court=doc.court,
                        court_name=doc.court_name,
                        year=doc.year,
                        title=doc.title,
                        date=doc.date,
                        parties=doc.parties,
                        url=doc.url,
                        text=(prefix + sub).strip(),
                        section_heading=heading,
                        chunk_index=idx,
                        citations=doc.citations,
                    )
                )
                idx += 1

    return chunks
