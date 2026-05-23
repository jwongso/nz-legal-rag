"""
Extract and link citations from secondary source chunks to the primary corpus.

Phase 2 of secondary source ingestion.

Handles two citation types:
  case        - [2024] NZCA 50  ->  NZCA/2024/50
  legislation - ERA 2000 s103A  ->  NZLEG/ERA2000/s103A

Run standalone against an already-ingested document:
    python -m ingest.secondary_citations --doc-id <uuid>
    python -m ingest.secondary_citations --all

Or call extract_and_link(conn, doc_id, chunk_id, text) from the ingest pipeline.
"""

import argparse
import re
import uuid
from dataclasses import dataclass

import psycopg2


# ---------------------------------------------------------------------------
# Case citation patterns
# ---------------------------------------------------------------------------

# Matches:  [2024] NZCA 50   [2019] NZEmpC 110   [2020] NZERA 243
_CASE_RE = re.compile(
    r"\[(\d{4})\]\s+(NZ[A-Za-z]+)\s+(\d+)"
)


def _normalise_case(year: str, court: str, number: str) -> str:
    return f"{court}/{year}/{number}"


# ---------------------------------------------------------------------------
# Legislation citation patterns
# ---------------------------------------------------------------------------

# Act name -> corpus abbreviation (NZLEG/<abbrev>/...)
_ACT_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Employment Relations Act[\s,]*2000", re.I), "ERA2000"),
    (re.compile(r"\bERA\b[\s,]*2000", re.I),                 "ERA2000"),
    (re.compile(r"\bERA\b",           re.I),                  "ERA2000"),  # fallback
    (re.compile(r"Residential Tenancies Act", re.I),           "RTA"),
    (re.compile(r"\bRTA\b",           re.I),                  "RTA"),
    (re.compile(r"Privacy Act[\s,]*2020", re.I),              "PA2020"),
    (re.compile(r"Companies Act[\s,]*1993", re.I),            "CA1993"),
    (re.compile(r"Crimes Act[\s,]*1961", re.I),               "CRA1961"),
    (re.compile(r"Contract and Commercial Law Act[\s,]*2017", re.I), "CCLA2017"),
]

# Section reference patterns: s103A  s 103A  section 103A  section103A
_SECTION_RE = re.compile(
    r"\b(?:s(?:ection)?\s*)(\d+[A-Z]?(?:\([a-z0-9]+\))*)",
    re.I,
)

# Window (chars) around an Act mention to look for section refs
_WINDOW = 300


def _extract_leg_citations(text: str) -> list[tuple[str, str]]:
    """Return list of (raw_text, normalised) for legislation citations."""
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for act_re, abbrev in _ACT_MAP:
        for m in act_re.finditer(text):
            # Look for section refs in a window around the Act mention
            window_start = max(0, m.start() - _WINDOW)
            window_end   = min(len(text), m.end() + _WINDOW)
            window = text[window_start:window_end]

            for sm in _SECTION_RE.finditer(window):
                section = "s" + sm.group(1)
                normalised = f"NZLEG/{abbrev}/{section}"
                if normalised in seen:
                    continue
                seen.add(normalised)
                raw = f"{m.group()} {sm.group()}"
                results.append((raw.strip(), normalised))

            # If no section found but Act clearly mentioned, record the Act root
            if not list(_SECTION_RE.finditer(window)):
                normalised = f"NZLEG/{abbrev}"
                if normalised not in seen:
                    seen.add(normalised)
                    results.append((m.group().strip(), normalised))

    return results


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@dataclass
class CitationRow:
    secondary_document_id: str
    secondary_chunk_id: str
    raw_citation: str
    normalised_citation: str
    citation_type: str          # case | legislation
    target_document_id: int | None
    confidence: float


def _lookup_primary(conn, normalised: str) -> int | None:
    """Return documents.id for exact citation match, or None."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE citation = %s LIMIT 1", (normalised,))
    row = cur.fetchone()
    return row[0] if row else None


def _insert_citations(conn, rows: list[CitationRow]) -> int:
    if not rows:
        return 0
    cur = conn.cursor()
    inserted = 0
    for r in rows:
        try:
            cur.execute("""
                INSERT INTO secondary_citations
                    (secondary_document_id, secondary_chunk_id, raw_citation,
                     normalised_citation, citation_type, target_document_id, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                r.secondary_document_id, r.secondary_chunk_id,
                r.raw_citation, r.normalised_citation,
                r.citation_type, r.target_document_id, r.confidence,
            ))
            inserted += cur.rowcount
        except Exception:
            conn.rollback()
            raise
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_and_link(conn, doc_id: str, chunk_id: str, text: str) -> list[CitationRow]:
    """Extract citations from one chunk and insert into secondary_citations."""
    rows: list[CitationRow] = []
    seen_norm: set[str] = set()

    # Case citations
    for m in _CASE_RE.finditer(text):
        year, court, number = m.group(1), m.group(2), m.group(3)
        normalised = _normalise_case(year, court, number)
        if normalised in seen_norm:
            continue
        seen_norm.add(normalised)
        target = _lookup_primary(conn, normalised)
        rows.append(CitationRow(
            secondary_document_id=doc_id,
            secondary_chunk_id=chunk_id,
            raw_citation=m.group().strip(),
            normalised_citation=normalised,
            citation_type="case",
            target_document_id=target,
            confidence=1.0,
        ))

    # Legislation citations
    for raw, normalised in _extract_leg_citations(text):
        if normalised in seen_norm:
            continue
        seen_norm.add(normalised)
        # Only link to corpus if it is a section-level citation
        target = _lookup_primary(conn, normalised) if "/" in normalised[7:] else None
        rows.append(CitationRow(
            secondary_document_id=doc_id,
            secondary_chunk_id=chunk_id,
            raw_citation=raw,
            normalised_citation=normalised,
            citation_type="legislation",
            target_document_id=target,
            confidence=0.9,
        ))

    _insert_citations(conn, rows)
    return rows


def process_document(conn, doc_id: str) -> dict:
    """Run citation extraction for all chunks of a document."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, text FROM secondary_chunks
        WHERE document_id = %s ORDER BY chunk_index
    """, (doc_id,))
    chunks = cur.fetchall()

    total = 0
    linked = 0
    for chunk_id, text in chunks:
        if not text:
            continue
        rows = extract_and_link(conn, doc_id, str(chunk_id), text or "")
        total  += len(rows)
        linked += sum(1 for r in rows if r.target_document_id is not None)

    cur.execute("""
        UPDATE secondary_documents SET updated_at = now() WHERE id = %s
    """, (doc_id,))
    conn.commit()

    return {"total_citations": total, "linked_to_corpus": linked}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract citations from secondary chunks")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--doc-id", help="UUID of a secondary_document to process")
    grp.add_argument("--all",    action="store_true", help="Process all embedded documents")
    args = parser.parse_args()

    conn = psycopg2.connect(dbname="nz_legal")

    if args.all:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title FROM secondary_documents
            WHERE parse_status = 'embedded'
        """)
        docs = cur.fetchall()
    else:
        cur = conn.cursor()
        cur.execute("SELECT id, title FROM secondary_documents WHERE id = %s", (args.doc_id,))
        docs = cur.fetchall()

    if not docs:
        print("No documents found.")
        return

    for doc_id, title in docs:
        print(f"\n[{title or doc_id}]")
        result = process_document(conn, str(doc_id))
        print(f"  citations found:      {result['total_citations']}")
        print(f"  linked to corpus:     {result['linked_to_corpus']}")

    conn.close()


if __name__ == "__main__":
    main()
