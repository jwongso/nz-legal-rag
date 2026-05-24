"""Secondary source pipeline status checker.

Queries PostgreSQL for pending/failed counts at each stage so a CI/CD runner
can decide which step to trigger next.

Exit codes:
  0  - pipeline is idle (nothing pending, no failures)
  1  - one or more stages have pending work or failures

Output modes:
  default  - human-readable table
  --json   - machine-readable JSON (for CI/CD parsing)
  --stage  - check a single stage only (ingest | review | analyze)
             exits 1 if that stage has pending work, 0 if idle

Run:
    python -m ingest.pipeline_status
    python -m ingest.pipeline_status --json
    python -m ingest.pipeline_status --stage review
"""

import argparse
import json
import sys

import psycopg2


def _query(conn) -> dict:
    cur = conn.cursor()
    cur.execute("""
        SELECT
            count(*) FILTER (WHERE parse_status = 'embedded')  AS ingest_ok,
            count(*) FILTER (WHERE parse_status = 'pending')   AS ingest_pending,
            count(*) FILTER (WHERE parse_status = 'failed')    AS ingest_failed,
            count(*) FILTER (WHERE parse_status = 'embedded'
                AND analysis_status = 'pending')               AS analyze_pending,
            count(*) FILTER (WHERE analysis_status = 'analyzed') AS analyze_ok,
            count(*) FILTER (WHERE analysis_status = 'failed')   AS analyze_failed
        FROM secondary_documents
    """)
    row = cur.fetchone()
    ingest_ok, ingest_pending, ingest_failed, \
        analyze_pending, analyze_ok, analyze_failed = row

    cur.execute("""
        SELECT
            count(*) FILTER (WHERE review_status = 'pending_llm')    AS review_pending,
            count(*) FILTER (WHERE review_status = 'auto_accepted'
                OR review_status LIKE 'llm_%')                       AS review_ok,
            count(*) FILTER (WHERE review_status = 'discarded')      AS discarded
        FROM secondary_citations
    """)
    row = cur.fetchone()
    review_pending, review_ok, discarded = row

    return {
        "ingest":  {"ok": ingest_ok,     "pending": ingest_pending, "failed": ingest_failed},
        "review":  {"ok": review_ok,     "pending": review_pending, "discarded": discarded},
        "analyze": {"ok": analyze_ok,    "pending": analyze_pending, "failed": analyze_failed},
    }


def _has_work(status: dict, stage: str | None) -> bool:
    stages = [stage] if stage else list(status.keys())
    for s in stages:
        d = status[s]
        if d.get("pending", 0) > 0 or d.get("failed", 0) > 0:
            return True
    return False


def _print_table(status: dict) -> None:
    print(f"{'Stage':<10} {'OK':>6} {'Pending':>9} {'Failed/Other':>13}")
    print("-" * 42)
    s = status["ingest"]
    print(f"{'ingest':<10} {s['ok']:>6} {s['pending']:>9} {s['failed']:>13}")
    s = status["review"]
    print(f"{'review':<10} {s['ok']:>6} {s['pending']:>9} {s['discarded']:>13}  (other=discarded)")
    s = status["analyze"]
    print(f"{'analyze':<10} {s['ok']:>6} {s['pending']:>9} {s['failed']:>13}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Secondary source pipeline status")
    parser.add_argument("--json",  action="store_true", help="Output JSON")
    parser.add_argument("--stage", choices=["ingest", "review", "analyze"],
                        help="Check a single stage; exits 1 if pending work exists")
    args = parser.parse_args()

    conn   = psycopg2.connect(dbname="nz_legal")
    status = _query(conn)
    conn.close()

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        _print_table(status)

    sys.exit(1 if _has_work(status, args.stage) else 0)


if __name__ == "__main__":
    main()
