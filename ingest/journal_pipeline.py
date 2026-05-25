"""
Scheduled journal ingestion pipeline.

Runs every 12 hours via systemd timer. Each run:
  1. Scrape journals for up to 10 new PDFs -> data/inbox/
  2. Ingest (parse, chunk, embed, cite)
  3. LLM citation review
  4. LLM document analysis
  5. Write report to data/journal_reports/YYYY-MM-DD_HHMM.txt

Run manually:
    python -m ingest.journal_pipeline
    python -m ingest.journal_pipeline --limit 5
    python -m ingest.journal_pipeline --dry-run
"""

import argparse
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

import config
from ingest.scrape_journals import fetch_new_pdfs

_REPORTS_DIR = config.DATA_DIR / "journal_reports"
_PYTHON      = sys.executable


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def _run_step(module: str, extra_args: list[str] = []) -> str:
    result = subprocess.run(
        [_PYTHON, "-m", module] + extra_args,
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent,
    )
    return (result.stdout + result.stderr).strip()


def _pipeline_status() -> dict:
    conn = psycopg2.connect(dbname="nz_legal")
    cur  = conn.cursor()
    cur.execute("""
        SELECT
            count(*) FILTER (WHERE parse_status = 'embedded')      AS ingest_ok,
            count(*) FILTER (WHERE parse_status = 'pending')        AS ingest_pending,
            count(*) FILTER (WHERE parse_status = 'failed')         AS ingest_failed,
            count(*) FILTER (WHERE parse_status = 'embedded'
                AND analysis_status = 'pending')                    AS analyze_pending,
            count(*) FILTER (WHERE analysis_status = 'analyzed')    AS analyze_ok,
            count(*) FILTER (WHERE analysis_status = 'failed')      AS analyze_failed
        FROM secondary_documents
    """)
    r = cur.fetchone()
    ingest_ok, ingest_pending, ingest_failed, \
        analyze_pending, analyze_ok, analyze_failed = r

    cur.execute("""
        SELECT
            count(*) FILTER (WHERE review_status = 'pending_llm')  AS review_pending,
            count(*) FILTER (WHERE review_status IN
                ('auto_accepted','llm_confirmed','llm_corrected'))  AS review_ok,
            count(*) FILTER (WHERE review_status = 'discarded')     AS discarded
        FROM secondary_citations
    """)
    review_pending, review_ok, discarded = cur.fetchone()
    conn.close()

    return {
        "ingest":  {"ok": ingest_ok,  "pending": ingest_pending,  "failed": ingest_failed},
        "review":  {"ok": review_ok,  "pending": review_pending,  "discarded": discarded},
        "analyze": {"ok": analyze_ok, "pending": analyze_pending, "failed": analyze_failed},
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _build_report(
    ts: str,
    fetched: list[dict],
    ingest_out: str,
    review_out: str,
    analyze_out: str,
    status: dict,
) -> str:
    sep   = "=" * 60
    lines = [sep, f"NZ Legal Journal Pipeline  {ts}", sep]

    # Fetch
    lines.append(f"\nFETCH  {len(fetched)} new articles downloaded")
    for f in fetched:
        title = textwrap.shorten(f.get("title", "untitled"), width=65)
        lines.append(f"  [{f.get('journal', '?')}] {title}")

    # Ingest
    chunks_total = sum(int(m) for m in re.findall(r"(\d+) chunks", ingest_out))
    files_done   = ingest_out.count("-> data/processed")
    cit_found    = sum(int(m) for m in re.findall(r"(\d+) found", ingest_out))
    cit_linked   = sum(int(m) for m in re.findall(r"(\d+) linked", ingest_out))
    lines.append(f"\nINGEST  {files_done} files, {chunks_total:,} chunks, "
                 f"{cit_found} citations ({cit_linked} linked to corpus)")

    # Review
    confirmed = _parse_count(review_out, "confirmed")
    corrected = _parse_count(review_out, "corrected")
    discarded = _parse_count(review_out, "discarded")
    lines.append(f"\nREVIEW  confirmed={confirmed}  corrected={corrected}  discarded={discarded}")

    # Analysis
    summaries = _parse_summaries(analyze_out)
    lines.append(f"\nANALYSIS  {len(summaries)} documents")
    for title, summary in summaries:
        lines.append(f"  [{textwrap.shorten(title, 50)}]")
        lines.append(f"    {textwrap.shorten(summary, 90)}")

    # Status
    st = status
    lines.append(f"\nPIPELINE STATUS")
    lines.append(f"  ingest:  {st['ingest']['ok']} ok | "
                 f"{st['ingest']['pending']} pending | {st['ingest']['failed']} failed")
    lines.append(f"  review:  {st['review']['ok']} ok | "
                 f"{st['review']['pending']} pending | {st['review']['discarded']} discarded")
    lines.append(f"  analyze: {st['analyze']['ok']} ok | "
                 f"{st['analyze']['pending']} pending | {st['analyze']['failed']} failed")
    lines.append(f"\nTotal secondary sources indexed: {st['ingest']['ok']}")
    lines.append(sep)
    return "\n".join(lines)


def _parse_count(text: str, label: str) -> int:
    m = re.search(rf"{label}:\s+(\d+)", text)
    return int(m.group(1)) if m else 0


def _parse_summaries(text: str) -> list[tuple[str, str]]:
    results = []
    current = ""
    for line in text.splitlines():
        m = re.match(r"\[(\d+/\d+)\]\s+(.+)", line)
        if m:
            current = m.group(2).strip()
        elif line.strip().startswith("Summary:") and current:
            summary = line.strip()[len("Summary:"):].strip()
            results.append((current, summary))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(limit: int = 10, dry_run: bool = False) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== Journal pipeline {ts} ===\n", flush=True)

    extra = ["--dry-run"] if dry_run else []

    print("Step 1/4: Fetching new articles ...", flush=True)
    fetched = fetch_new_pdfs(limit=limit, dry_run=dry_run)
    print(f"  {len(fetched)} {'would be ' if dry_run else ''}downloaded\n", flush=True)

    if not fetched and not dry_run:
        print("Nothing new to process.")
        return

    print("Step 2/4: Ingesting ...", flush=True)
    ingest_out = _run_step("ingest.ingest_secondary")
    print(ingest_out, flush=True)

    print("\nStep 3/4: LLM citation review ...", flush=True)
    review_out = _run_step("ingest.review_citations", extra)
    print(review_out, flush=True)

    print("\nStep 4/4: LLM document analysis ...", flush=True)
    analyze_out = _run_step("ingest.analyze_secondary", extra)
    print(analyze_out, flush=True)

    status = _pipeline_status()
    report = _build_report(ts, fetched, ingest_out, review_out, analyze_out, status)

    if not dry_run:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        slug     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        out_path = _REPORTS_DIR / f"report_{slug}.txt"
        out_path.write_text(report)
        print(f"\nReport saved: {out_path}", flush=True)

    print("\n" + report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled journal ingestion pipeline")
    parser.add_argument("--limit",   type=int, default=10, help="Max PDFs per run")
    parser.add_argument("--dry-run", action="store_true",  help="No downloads or DB writes")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
