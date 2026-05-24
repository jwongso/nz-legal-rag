"""
Extract and link citations from secondary source chunks to the primary corpus.

Phase 2 of secondary source ingestion.

Handles two citation types:
  case        - [2024] NZCA 50  ->  NZCA/2024/50
  legislation - ERA 2000 s103A  ->  NZLEG/ERA2000/s103A

Confidence scoring:
  Case citations
    Known NZ court code + exact pattern    -> 1.00
    Unknown code (reporters: NZLR, NZLJ)  -> 0.60  [pending_llm]

  Legislation citations
    Full Act name + section within 50 chars  -> 0.95
    Full Act name + section within 300 chars -> 0.80
    Short abbreviation + section             -> 0.65  [pending_llm]
    Act name only, no section               -> 0.45  [pending_llm]

Citations with confidence < REVIEW_THRESHOLD are flagged review_status='pending_llm'
for the LLM review pass (review_citations.py). Others are 'auto_accepted'.

Run standalone:
    python -m ingest.secondary_citations --doc-id <uuid>
    python -m ingest.secondary_citations --all
"""

import argparse
import re
from dataclasses import dataclass

import psycopg2


REVIEW_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Known NZ court codes (anything else in a [YEAR] X NNN citation is a reporter)
# ---------------------------------------------------------------------------

_KNOWN_COURTS = {
    "NZSC", "NZCA", "NZHC", "NZDC", "NZEmpC", "NZERA",
    "NZFC", "NZEnvC", "NZACC", "NZCorC", "NZLCDT",
    "NZHRRT", "NZREADT", "NZTT",
}

# Matches:  [2024] NZCA 50   [2019] NZEmpC 110
_CASE_RE = re.compile(r"\[(\d{4})\]\s+(NZ[A-Za-z]+)\s+(\d+)")


def _case_confidence(court: str) -> float:
    return 1.0 if court in _KNOWN_COURTS else 0.60


# ---------------------------------------------------------------------------
# Legislation patterns
# ---------------------------------------------------------------------------

# (pattern, abbrev, match_type)
#   match_type "full" = full Act name  -> higher base confidence
#   match_type "abbr" = short abbreviation -> lower base confidence
_ACT_MAP: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"Employment Relations Act[\s,]*2000", re.I), "ERA2000", "full"),
    (re.compile(r"\bERA\b[\s,]*2000",                 re.I), "ERA2000", "full"),
    (re.compile(r"\bERA\b",                           re.I), "ERA2000", "abbr"),
    (re.compile(r"Residential Tenancies Act",          re.I), "RTA",     "full"),
    (re.compile(r"\bRTA\b",                           re.I), "RTA",     "abbr"),
    (re.compile(r"Privacy Act[\s,]*2020",             re.I), "PA2020",  "full"),
    (re.compile(r"Companies Act[\s,]*1993",           re.I), "CA1993",  "full"),
    (re.compile(r"Crimes Act[\s,]*1961",              re.I), "CRA1961", "full"),
    (re.compile(r"Contract and Commercial Law Act[\s,]*2017", re.I), "CCLA2017", "full"),
]

_SECTION_RE = re.compile(
    r"\b(?:s(?:ection)?\s*)(\d+[A-Z]?(?:\([a-z0-9]+\))*)",
    re.I,
)

_WINDOW = 300
_TIGHT_WINDOW = 50  # chars - section right next to Act name


def _leg_confidence(match_type: str, has_section: bool, section_distance: int) -> float:
    if not has_section:
        return 0.45  # Act mention only, no section - very ambiguous
    if match_type == "abbr":
        return 0.65  # short abbreviation - could be false positive
    # Full Act name + section
    if section_distance <= _TIGHT_WINDOW:
        return 0.95
    return 0.80


def _extract_leg_citations(text: str) -> list[tuple[str, str, float]]:
    """Return (raw, normalised, confidence) for legislation citations."""
    results: list[tuple[str, str, float]] = []
    seen: set[str] = set()

    for act_re, abbrev, match_type in _ACT_MAP:
        for m in act_re.finditer(text):
            window_start = max(0, m.start() - _WINDOW)
            window_end   = min(len(text), m.end() + _WINDOW)
            window       = text[window_start:window_end]
            m_pos_in_window = m.start() - window_start

            found_section = False
            for sm in _SECTION_RE.finditer(window):
                section = "s" + sm.group(1)
                normalised = f"NZLEG/{abbrev}/{section}"
                if normalised in seen:
                    continue
                seen.add(normalised)
                found_section = True
                distance = abs(sm.start() - m_pos_in_window)
                conf = _leg_confidence(match_type, True, distance)
                raw = f"{m.group()} {sm.group()}"
                results.append((raw.strip(), normalised, conf))

            if not found_section:
                normalised = f"NZLEG/{abbrev}"
                if normalised not in seen:
                    seen.add(normalised)
                    conf = _leg_confidence(match_type, False, 0)
                    results.append((m.group().strip(), normalised, conf))

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
    citation_type: str
    target_document_id: int | None
    confidence: float
    review_status: str   # auto_accepted | pending_llm


def _lookup_primary(conn, normalised: str) -> int | None:
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
                     normalised_citation, citation_type, target_document_id,
                     confidence, review_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                r.secondary_document_id, r.secondary_chunk_id,
                r.raw_citation, r.normalised_citation,
                r.citation_type, r.target_document_id,
                r.confidence, r.review_status,
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
    """Extract citations from one chunk, score confidence, insert into DB."""
    rows: list[CitationRow] = []
    seen_norm: set[str] = set()

    # Case citations
    for m in _CASE_RE.finditer(text):
        year, court, number = m.group(1), m.group(2), m.group(3)
        normalised = f"{court}/{year}/{number}"
        if normalised in seen_norm:
            continue
        seen_norm.add(normalised)
        conf = _case_confidence(court)
        rows.append(CitationRow(
            secondary_document_id=doc_id,
            secondary_chunk_id=chunk_id,
            raw_citation=m.group().strip(),
            normalised_citation=normalised,
            citation_type="case",
            target_document_id=_lookup_primary(conn, normalised),
            confidence=conf,
            review_status="auto_accepted" if conf >= REVIEW_THRESHOLD else "pending_llm",
        ))

    # Legislation citations
    for raw, normalised, conf in _extract_leg_citations(text):
        if normalised in seen_norm:
            continue
        seen_norm.add(normalised)
        target = _lookup_primary(conn, normalised) if normalised.count("/") >= 2 else None
        rows.append(CitationRow(
            secondary_document_id=doc_id,
            secondary_chunk_id=chunk_id,
            raw_citation=raw,
            normalised_citation=normalised,
            citation_type="legislation",
            target_document_id=target,
            confidence=conf,
            review_status="auto_accepted" if conf >= REVIEW_THRESHOLD else "pending_llm",
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

    total = linked = pending = 0
    for chunk_id, text in chunks:
        if not text:
            continue
        rows = extract_and_link(conn, doc_id, str(chunk_id), text)
        total   += len(rows)
        linked  += sum(1 for r in rows if r.target_document_id is not None)
        pending += sum(1 for r in rows if r.review_status == "pending_llm")

    cur.execute("UPDATE secondary_documents SET updated_at = now() WHERE id = %s", (doc_id,))
    conn.commit()
    return {"total_citations": total, "linked_to_corpus": linked, "pending_llm": pending}


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
        cur.execute("SELECT id, title FROM secondary_documents WHERE parse_status = 'embedded'")
        docs = cur.fetchall()
    else:
        cur = conn.cursor()
        cur.execute("SELECT id, title FROM secondary_documents WHERE id = %s", (args.doc_id,))
        docs = cur.fetchall()

    for doc_id, title in docs:
        print(f"\n[{title or doc_id}]")
        r = process_document(conn, str(doc_id))
        print(f"  citations:    {r['total_citations']}")
        print(f"  linked:       {r['linked_to_corpus']}")
        print(f"  pending LLM:  {r['pending_llm']}")

    conn.close()


if __name__ == "__main__":
    main()
