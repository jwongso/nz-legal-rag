"""LLM review pass for low-confidence citation extractions.

Fetches all secondary_citations with review_status='pending_llm', sends each
to the LLM with its source chunk for context, and records the verdict:

  CONFIRM   - extraction is correct as-is
  CORRECT   - extraction is wrong; LLM provides the right normalised form
  DISCARD   - not a real citation (false positive)

Updates secondary_citations with:
  review_status      -> llm_confirmed | llm_corrected | discarded
  llm_verdict        -> raw LLM response (truncated)
  llm_corrected_citation -> corrected normalised form (if CORRECT)
  reviewed_at        -> now()

Run:
    python -m ingest.review_citations
    python -m ingest.review_citations --dry-run       # print prompts, no DB writes
    python -m ingest.review_citations --doc-id <uuid> # one document only
"""

import argparse
import json
import re
import textwrap
from datetime import datetime, timezone

import httpx
import psycopg2

import config

_REVIEW_PROMPT = """\
You are a legal citation auditor for New Zealand law.

You will be given:
1. A short passage from a law review article
2. A citation that was automatically extracted from that passage
3. The normalised form the system assigned to it

Your job is to verify the extraction and respond with exactly one of:

CONFIRM
  The citation is correctly identified and the normalised form is right.

CORRECT: <corrected-form>
  The citation exists in the passage but the normalised form is wrong.
  Provide the corrected normalised form using the pattern:
    Case:        COURT/YEAR/NUMBER   e.g. NZCA/2024/50
    Legislation: NZLEG/ABBREV/sN    e.g. NZLEG/ERA2000/s103A

DISCARD
  This is not a real NZ legal citation (e.g. a law reporter reference like
  NZLR or NZLJ, a foreign case, or a false positive from the abbreviation).

Respond with ONLY one of the above - no explanation.

---
PASSAGE:
{passage}

---
EXTRACTED RAW:    {raw}
NORMALISED FORM:  {normalised}
CITATION TYPE:    {ctype}
"""

_VERDICT_RE = re.compile(
    r"^\s*(CONFIRM|DISCARD|CORRECT\s*:\s*(\S+))",
    re.IGNORECASE,
)


def _call_llm(prompt: str, base_url: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(base_url=base_url, timeout=60) as client:
        resp = client.post("/chat/completions", json=payload)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_verdict(raw: str) -> tuple[str, str | None]:
    """Return (status, corrected_citation_or_None)."""
    m = _VERDICT_RE.match(raw)
    if not m:
        return "llm_confirmed", None   # fallback: treat ambiguous as confirm
    keyword = m.group(1).upper()
    if keyword.startswith("CORRECT"):
        corrected = m.group(2)
        return "llm_corrected", corrected
    if keyword == "DISCARD":
        return "discarded", None
    return "llm_confirmed", None


def _fetch_chunk_text(conn, chunk_id: str) -> str:
    cur = conn.cursor()
    cur.execute("SELECT text FROM secondary_chunks WHERE id = %s", (chunk_id,))
    row = cur.fetchone()
    return row[0] if row else ""


def _fetch_pending(conn, doc_id: str | None) -> list[dict]:
    cur = conn.cursor()
    if doc_id:
        cur.execute("""
            SELECT sc.id, sc.raw_citation, sc.normalised_citation,
                   sc.citation_type, sc.secondary_chunk_id, sc.confidence
            FROM secondary_citations sc
            JOIN secondary_chunks ch ON ch.id = sc.secondary_chunk_id
            JOIN secondary_documents d ON d.id = sc.secondary_document_id
            WHERE sc.review_status = 'pending_llm'
              AND sc.secondary_document_id = %s
            ORDER BY sc.confidence, sc.id
        """, (doc_id,))
    else:
        cur.execute("""
            SELECT sc.id, sc.raw_citation, sc.normalised_citation,
                   sc.citation_type, sc.secondary_chunk_id, sc.confidence
            FROM secondary_citations sc
            WHERE sc.review_status = 'pending_llm'
            ORDER BY sc.confidence, sc.id
        """)
    cols = ["id", "raw_citation", "normalised_citation", "citation_type",
            "secondary_chunk_id", "confidence"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _write_verdict(conn, cit_id: str, status: str, verdict_text: str,
                   corrected: str | None) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE secondary_citations SET
            review_status          = %s,
            llm_verdict            = %s,
            llm_corrected_citation = %s,
            reviewed_at            = %s
        WHERE id = %s
    """, (status, verdict_text[:500], corrected, datetime.now(timezone.utc), cit_id))
    conn.commit()


def run(args: argparse.Namespace) -> None:
    conn = psycopg2.connect(dbname="nz_legal")
    pending = _fetch_pending(conn, args.doc_id)

    if not pending:
        print("No citations pending LLM review.")
        return

    base_url = args.base_url or config.LLM_BASE_URL
    model    = args.model    or config.LLM_MODEL

    print(f"LLM citation review")
    print(f"  pending:  {len(pending)}")
    print(f"  model:    {model}")
    print(f"  base_url: {base_url}")
    print(f"  dry_run:  {args.dry_run}")
    print()

    counts = {"llm_confirmed": 0, "llm_corrected": 0, "discarded": 0, "error": 0}

    for i, cit in enumerate(pending, 1):
        chunk_text = _fetch_chunk_text(conn, cit["secondary_chunk_id"])
        passage = textwrap.shorten(chunk_text, width=800, placeholder="...")

        prompt = _REVIEW_PROMPT.format(
            passage=passage,
            raw=cit["raw_citation"],
            normalised=cit["normalised_citation"],
            ctype=cit["citation_type"],
        )

        print(f"[{i}/{len(pending)}] {cit['raw_citation']!r} -> {cit['normalised_citation']}"
              f"  (conf={cit['confidence']:.2f})")

        if args.dry_run:
            print(f"  [dry-run] prompt ready ({len(prompt)} chars)")
            continue

        try:
            raw_response = _call_llm(prompt, base_url, model)
            status, corrected = _parse_verdict(raw_response)
            counts[status] += 1
            _write_verdict(conn, cit["id"], status, raw_response, corrected)
            label = f"CORRECTED -> {corrected}" if corrected else status.upper()
            print(f"  {label}")
        except Exception as e:
            print(f"  [error] {e}")
            counts["error"] += 1

    conn.close()
    print()
    print(f"Done.")
    print(f"  confirmed:  {counts['llm_confirmed']}")
    print(f"  corrected:  {counts['llm_corrected']}")
    print(f"  discarded:  {counts['discarded']}")
    if counts["error"]:
        print(f"  errors:     {counts['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM review of low-confidence citations")
    parser.add_argument("--doc-id",   default=None, help="Review one document only")
    parser.add_argument("--dry-run",  action="store_true", help="Print prompts, no DB writes")
    parser.add_argument("--model",    default=None, help="LLM model (default: config.LLM_MODEL)")
    parser.add_argument("--base-url", default=None, help="LLM base URL (default: config.LLM_BASE_URL)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
