"""LLM analysis pass for ingested secondary source documents.

Runs AFTER ingest_secondary.py and review_citations.py. Produces a structured
research brief for each document: summary, key legal issues, confirmed cases
and statutes, and practical relevance notes.

Results are stored in secondary_documents:
  summary        - 2-3 sentence plain-English summary
  key_issues     - array of key legal questions the article addresses
  analysis_text  - full structured brief (Markdown)
  analysis_status -> analyzed | failed

Intentionally separate from ingestion so it can be run when LLM resources
are available without re-embedding anything.

Run:
    python -m ingest.analyze_secondary               # all pending documents
    python -m ingest.analyze_secondary --doc-id <uuid>
    python -m ingest.analyze_secondary --dry-run     # show context only
"""

import argparse
import textwrap
from datetime import datetime, timezone

import httpx
import psycopg2

import config


_ANALYSIS_PROMPT = """\
You are a legal research assistant specialising in New Zealand law.

Read the following extracts from a law review article and the citations found \
inside it, then produce a structured research brief.

Respond in this exact format (use the headings as shown):

## Summary
2-3 sentences. What is the article's core argument or finding?

## Key Legal Issues
Bullet list. What specific legal questions does the article address?

## Cases Discussed
Bullet list. For each confirmed case citation, one line: case name and \
what point the article makes about it. If no cases, write "None confirmed."

## Statutes Analysed
Bullet list. Which Acts or sections does the article focus on? \
If none, write "None identified."

## Practical Relevance
1-2 sentences. How is this article useful for NZ legal research on the \
topics it covers?

---
ARTICLE TITLE: {title}

CONFIRMED CITATIONS:
{citations}

ARTICLE EXTRACTS (first sections):
{extracts}
"""


def _fetch_pending(conn, doc_id: str | None) -> list[dict]:
    cur = conn.cursor()
    if doc_id:
        cur.execute("""
            SELECT id, title, source_type
            FROM secondary_documents
            WHERE id = %s AND parse_status = 'embedded'
        """, (doc_id,))
    else:
        cur.execute("""
            SELECT id, title, source_type
            FROM secondary_documents
            WHERE parse_status = 'embedded'
              AND analysis_status = 'pending'
            ORDER BY created_at
        """)
    cols = ["id", "title", "source_type"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetch_context(conn, doc_id: str) -> tuple[str, str]:
    """Return (citations_block, extracts_block) for the document."""
    cur = conn.cursor()

    # Confirmed citations only (auto_accepted or llm_confirmed or llm_corrected)
    cur.execute("""
        SELECT DISTINCT
            COALESCE(llm_corrected_citation, normalised_citation) AS citation,
            raw_citation,
            citation_type,
            review_status
        FROM secondary_citations
        WHERE secondary_document_id = %s
          AND review_status IN ('auto_accepted', 'llm_confirmed', 'llm_corrected')
        ORDER BY citation_type, citation
    """, (doc_id,))
    cit_rows = cur.fetchall()

    if cit_rows:
        lines = []
        for norm, raw, ctype, status in cit_rows:
            tag = "(corrected)" if status == "llm_corrected" else ""
            lines.append(f"  [{ctype}] {raw} -> {norm} {tag}".strip())
        citations_block = "\n".join(lines)
    else:
        citations_block = "  No confirmed citations."

    # First 6 chunks (abstract + intro) for context - not the whole document
    cur.execute("""
        SELECT text FROM secondary_chunks
        WHERE document_id = %s
        ORDER BY chunk_index
        LIMIT 6
    """, (doc_id,))
    chunks = [row[0] for row in cur.fetchall() if row[0]]
    extracts_block = "\n\n---\n\n".join(
        textwrap.shorten(c, width=600, placeholder="...") for c in chunks
    )

    return citations_block, extracts_block


def _call_llm(prompt: str, base_url: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(base_url=base_url, timeout=120) as client:
        resp = client.post("/chat/completions", json=payload)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_analysis(text: str) -> tuple[str, list[str]]:
    """Extract summary and key_issues list from the structured response."""
    summary = ""
    issues: list[str] = []

    lines = text.splitlines()
    in_summary = False
    in_issues = False

    for line in lines:
        if line.strip().startswith("## Summary"):
            in_summary = True
            in_issues = False
            continue
        if line.strip().startswith("## Key Legal Issues"):
            in_summary = False
            in_issues = True
            continue
        if line.strip().startswith("## "):
            in_summary = False
            in_issues = False
            continue

        if in_summary and line.strip():
            summary = (summary + " " + line.strip()).strip()
        if in_issues and line.strip().startswith("-"):
            issues.append(line.strip().lstrip("- ").strip())

    return summary[:1000], issues[:10]


def _write_result(conn, doc_id: str, analysis_text: str,
                  summary: str, key_issues: list[str]) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE secondary_documents SET
            analysis_status = 'analyzed',
            summary         = %s,
            key_issues      = %s,
            analysis_text   = %s,
            analyzed_at     = %s
        WHERE id = %s
    """, (summary, key_issues, analysis_text,
          datetime.now(timezone.utc), doc_id))
    conn.commit()


def _mark_failed(conn, doc_id: str, error: str) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE secondary_documents SET
            analysis_status = 'failed',
            analysis_text   = %s
        WHERE id = %s
    """, (error[:500], doc_id))
    conn.commit()


def run(args: argparse.Namespace) -> None:
    conn    = psycopg2.connect(dbname="nz_legal")
    docs    = _fetch_pending(conn, args.doc_id)
    base_url = args.base_url or config.LLM_BASE_URL
    model   = args.model    or config.LLM_MODEL

    if not docs:
        print("No documents pending analysis.")
        return

    print(f"LLM document analysis")
    print(f"  documents: {len(docs)}")
    print(f"  model:     {model}")
    print(f"  dry_run:   {args.dry_run}")
    print()

    for i, doc in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {doc['title'] or doc['id']}")
        citations_block, extracts_block = _fetch_context(conn, doc["id"])

        prompt = _ANALYSIS_PROMPT.format(
            title=doc["title"] or "(untitled)",
            citations=citations_block,
            extracts=extracts_block,
        )

        if args.dry_run:
            print(f"  Context: {len(citations_block.splitlines())} citation lines, "
                  f"{len(extracts_block)} chars of extracts")
            print(f"  Prompt length: {len(prompt)} chars")
            print()
            continue

        try:
            analysis_text = _call_llm(prompt, base_url, model)
            summary, key_issues = _parse_analysis(analysis_text)
            _write_result(conn, doc["id"], analysis_text, summary, key_issues)

            print(f"  Summary: {summary[:120]}...")
            print(f"  Key issues: {len(key_issues)}")
            print()
        except Exception as e:
            print(f"  [error] {e}")
            _mark_failed(conn, doc["id"], str(e))

    conn.close()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM analysis of ingested secondary documents")
    parser.add_argument("--doc-id",   default=None,
                        help="Analyse one document only (by UUID)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show context without calling LLM")
    parser.add_argument("--model",    default=None,
                        help="LLM model (default: config.LLM_MODEL)")
    parser.add_argument("--base-url", default=None,
                        help="LLM base URL (default: config.LLM_BASE_URL)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
