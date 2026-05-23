"""
Chunk secondary source text (journal articles, memos, commentary).

Strategy:
  1. Split on recognisable section headings (numbered or title-case lines).
  2. Within each section, apply a sliding word window identical to the primary chunker.
  3. Tag each chunk with a chunk_type heuristic (abstract, footnote, conclusion, body).

The output is intentionally close to the primary Chunk dataclass so the same
Qdrant upsert code can handle both, but it is a separate dataclass to keep
primary and secondary schemas independent.
"""

import re
from dataclasses import dataclass, field

import config


@dataclass
class SecondaryChunk:
    doc_id: str              # secondary_documents.id (UUID string)
    chunk_index: int
    section_title: str
    chunk_type: str          # abstract | footnote | conclusion | body
    text: str
    token_count: int


_SECTION_RE = re.compile(
    r"^("
    r"\d+\.\s+[A-Z][a-z]"                      # "1. Introduction" (digit + period required)
    r"|[IVXLC]+\.\s+[A-Z][a-z]"                # "IV. Analysis"
    r"|[A-Z][A-Z\s]{3,40}$"                    # "BACKGROUND", "THE FACTS" (all-caps headings)
    r"|(?:Abstract|Introduction|Background|Facts|Issues?|Analysis|"
    r"Conclusion|Summary|Methodology)\b"        # known heading words
    r")",
    re.MULTILINE | re.IGNORECASE,
)

_ABSTRACT_RE = re.compile(r"\babstract\b", re.I)
_CONCLUSION_RE = re.compile(r"\b(conclusion|summary)\b", re.I)
_FOOTNOTE_RE = re.compile(r"^\s*\d{1,3}[\.\)]\s", re.MULTILINE)


def _chunk_type(heading: str, body: str) -> str:
    if _ABSTRACT_RE.search(heading):
        return "abstract"
    if _CONCLUSION_RE.search(heading):
        return "conclusion"
    # Heuristic: body dominated by short numbered lines -> footnotes
    footnote_lines = len(_FOOTNOTE_RE.findall(body))
    if footnote_lines > 5 and footnote_lines / max(1, body.count("\n")) > 0.4:
        return "footnote"
    return "body"


def _word_count(text: str) -> int:
    return len(text.split())


def _sliding_window(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + max_words]))
        i += max_words - overlap
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs. Empty heading = preamble."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("", text.strip())]

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        # Extend match to end of its line so heading is complete
        line_end = text.find("\n", m.start())
        if line_end == -1:
            line_end = len(text)
        heading = text[m.start(): line_end].strip()
        body_start = line_end + 1
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((heading, body))

    return sections


def chunk_secondary(doc_id: str, text: str) -> list[SecondaryChunk]:
    sections = _split_sections(text)
    chunks: list[SecondaryChunk] = []
    idx = 0

    for heading, body in sections:
        ctype = _chunk_type(heading, body)
        if _word_count(body) <= config.CHUNK_SIZE:
            combined = f"{heading}\n\n{body}".strip() if heading else body
            if _word_count(combined) < config.CHUNK_MIN_WORDS:
                continue
            chunks.append(SecondaryChunk(
                doc_id=doc_id,
                chunk_index=idx,
                section_title=heading,
                chunk_type=ctype,
                text=combined,
                token_count=_word_count(combined),
            ))
            idx += 1
        else:
            for sub in _sliding_window(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                if _word_count(sub) < config.CHUNK_MIN_WORDS:
                    continue
                prefix = f"{heading}\n\n" if heading else ""
                chunks.append(SecondaryChunk(
                    doc_id=doc_id,
                    chunk_index=idx,
                    section_title=heading,
                    chunk_type=ctype,
                    text=(prefix + sub).strip(),
                    token_count=_word_count(prefix + sub),
                ))
                idx += 1

    return chunks
