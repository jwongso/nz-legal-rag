"""
Migrate existing Qdrant data into PostgreSQL.

Reads all points from Qdrant and populates:
  documents        - one row per case
  chunks           - one row per chunk (with text for BM25 full-text search)
  citations        - citation strings extracted at ingest time
  sentencing_cases - structured sentencing fields (where extracted)
  employment_cases - structured employment grievance fields (where extracted)
  ingest_runs      - one row per (court, year) pair from progress file

Safe to re-run - all inserts use ON CONFLICT DO NOTHING or DO UPDATE.

Usage:
    python -m db.migrate_from_qdrant
    python -m db.migrate_from_qdrant --dry-run
    python -m db.migrate_from_qdrant --court NZCA
    python -m db.migrate_from_qdrant --no-text   # skip storing chunk text (smaller DB)
"""

import argparse
import json
import re
import time
from datetime import date, datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

import config

_PROGRESS_FILE = Path("data/ingest_progress.json")

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_date(date_str: str) -> date | None:
    if not date_str:
        return None
    m = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", date_str)
    if not m:
        return None
    try:
        day = int(m.group(1))
        month = _MONTH_MAP.get(m.group(2).lower())
        year = int(m.group(3))
        if month:
            return date(year, month, day)
    except (ValueError, TypeError):
        pass
    return None


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(dbname="nz_legal")


def _migrate_ingest_runs(cur: psycopg2.extensions.cursor) -> None:
    if not _PROGRESS_FILE.exists():
        print("  No progress file found - skipping ingest_runs.")
        return
    data = json.loads(_PROGRESS_FILE.read_text())
    completed = data.get("completed", [])
    log = {entry["key"]: entry for entry in data.get("log", [])}

    rows = []
    for key in completed:
        parts = key.split(":")
        if len(parts) != 2:
            continue
        court, year_str = parts
        entry = log.get(key, {})
        rows.append((
            court,
            int(year_str),
            "completed",
            entry.get("chunks"),
            entry.get("at"),
        ))

    psycopg2.extras.execute_batch(cur, """
        INSERT INTO ingest_runs (court_code, year, status, chunks_indexed, completed_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (court_code, year) DO UPDATE
            SET status = EXCLUDED.status,
                chunks_indexed = EXCLUDED.chunks_indexed,
                completed_at = EXCLUDED.completed_at
    """, rows)
    print(f"  Ingest runs: {len(rows)} rows upserted.")


def run(
    dry_run: bool = False,
    court_filter: str | None = None,
    store_text: bool = True,
) -> None:
    qdrant = QdrantClient(url=config.QDRANT_URL)

    scroll_filter = None
    if court_filter:
        scroll_filter = Filter(must=[
            FieldCondition(key="court", match=MatchValue(value=court_filter))
        ])

    conn = _connect() if not dry_run else None
    cur  = conn.cursor() if conn else None

    # Track which case_ids have already been inserted this run
    seen_cases: dict[str, int] = {}   # case_id -> documents.id

    batch_size  = 200
    offset      = None
    total       = 0
    doc_count   = 0
    chunk_count = 0
    sent_count  = 0
    emp_count   = 0
    cite_count  = 0
    t0 = time.time()

    print(f"Starting migration {'(dry run) ' if dry_run else ''}from Qdrant -> PostgreSQL")
    if court_filter:
        print(f"  Filtering to court: {court_filter}")

    while True:
        results, next_offset = qdrant.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            with_payload=True,
            limit=batch_size,
            offset=offset,
        )
        if not results:
            break

        doc_rows   = []
        chunk_rows = []
        sent_rows  = []
        emp_rows   = []
        cite_rows  = []

        for r in results:
            p = r.payload
            case_id    = p.get("case_id", "")
            court      = p.get("court", "")
            year       = p.get("year")
            chunk_idx  = p.get("chunk_index", 0)

            # --- documents ---
            if case_id not in seen_cases:
                decision_date = _parse_date(p.get("date", ""))
                doc_rows.append((
                    p.get("title"),
                    case_id,
                    court,
                    decision_date,
                    p.get("url"),
                    "decision",
                    "NZ",
                    p.get("doc_hash"),
                    "completed",
                ))
                seen_cases[case_id] = None  # placeholder until we get the real id

            # --- chunks ---
            chunk_rows.append((
                case_id,
                chunk_idx,
                p.get("section_heading"),
                p.get("text") if store_text else None,
                len(p.get("text") or "") // 4,  # approx token count
                str(r.id),  # qdrant_point_id
            ))

            # --- citations ---
            for c in (p.get("citations") or []):
                cite_rows.append((case_id, c))

            # --- sentencing ---
            sent = p.get("sentencing") or {}
            if isinstance(sent, dict) and sent.get("has_data"):
                sent_rows.append((
                    case_id,
                    sent.get("sentence_type"),
                    sent.get("starting_point_months"),
                    sent.get("final_sentence_months"),
                    sent.get("home_detention_months"),
                    sent.get("community_work_hours"),
                    sent.get("guilty_plea_discount_pct"),
                    sent.get("has_guilty_plea"),
                    sent.get("has_remorse"),
                    sent.get("has_previous_convictions"),
                    bool(sent.get("flag_self_defence")),
                    bool(sent.get("flag_provocation")),
                    bool(sent.get("flag_mental_health")),
                    bool(sent.get("flag_intoxication")),
                    bool(sent.get("flag_youth")),
                    bool(sent.get("flag_tikanga_maori")),
                    bool(sent.get("flag_cultural_factors")),
                ))

            # --- employment/PG ---
            pg = p.get("pg") or {}
            if isinstance(pg, dict) and pg.get("has_data"):
                emp_rows.append((
                    case_id,
                    (pg.get("grievance_types") or [None])[0],
                    pg.get("grievance_types") or [],
                    pg.get("reinstatement_ordered"),
                    pg.get("contributory_conduct_pct"),
                ))

        if not dry_run and doc_rows:
            # Insert documents, get back ids
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO documents
                    (title, citation, court, decision_date, source_url,
                     document_type, jurisdiction, checksum, ingestion_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (citation) DO UPDATE
                    SET title = EXCLUDED.title,
                        decision_date = COALESCE(EXCLUDED.decision_date, documents.decision_date),
                        updated_at = NOW()
            """, doc_rows)

            # Resolve case_id -> documents.id for this batch
            citations_in_batch = [r[1] for r in doc_rows]
            cur.execute(
                "SELECT id, citation FROM documents WHERE citation = ANY(%s)",
                (citations_in_batch,)
            )
            for doc_id, citation in cur.fetchall():
                seen_cases[citation] = doc_id

            doc_count += len(doc_rows)

        if not dry_run and chunk_rows:
            # Resolve case_id to document_id for chunks
            resolved_chunks = []
            for case_id, chunk_idx, section, text, tokens, qid in chunk_rows:
                doc_id = seen_cases.get(case_id)
                if doc_id:
                    clean_text = text.replace('\x00', '') if text else text
                    resolved_chunks.append((doc_id, chunk_idx, section, clean_text, tokens, qid))

            psycopg2.extras.execute_batch(cur, """
                INSERT INTO chunks
                    (document_id, chunk_index, section_title, text, token_count, qdrant_point_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, chunk_index) DO NOTHING
            """, resolved_chunks)
            chunk_count += len(resolved_chunks)

        if not dry_run and cite_rows:
            # Insert citations (to_document_id resolved later via resolve_citations.sql)
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO citations (from_document_id, cited_text)
                SELECT d.id, %s
                FROM documents d WHERE d.citation = %s
                ON CONFLICT (from_document_id, cited_text) DO NOTHING
            """, [(c, cid) for cid, c in cite_rows])
            cite_count += len(cite_rows)

        if not dry_run and sent_rows:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO sentencing_cases
                    (document_id, offence, starting_point, final_sentence,
                     home_detention_months, community_work_hours, guilty_plea_discount,
                     flag_self_defence, flag_provocation, flag_mental_health,
                     flag_intoxication, flag_youth, flag_tikanga_maori, flag_cultural_factors,
                     flag_previous_convictions)
                SELECT d.id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                FROM documents d WHERE d.citation = %s
                ON CONFLICT (document_id) DO UPDATE
                    SET offence = EXCLUDED.offence,
                        starting_point = EXCLUDED.starting_point,
                        final_sentence = EXCLUDED.final_sentence
            """, [
                (st, sp, fs, hd, cw, gp, sd, prov, mh, intox, youth, tik, cult, prev, cid)
                for cid, st, sp, fs, hd, cw, gp, hgp, rem, prev, sd, prov, mh, intox, youth, tik, cult in sent_rows
            ])
            sent_count += len(sent_rows)

        if not dry_run and emp_rows:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO employment_cases
                    (document_id, grievance_type, grievance_types, reinstatement, contributory_conduct_pct)
                SELECT d.id, %s, %s, %s, %s
                FROM documents d WHERE d.citation = %s
                ON CONFLICT (document_id) DO NOTHING
            """, [(gt, gts, ri, cc, cid) for cid, gt, gts, ri, cc in emp_rows])
            emp_count += len(emp_rows)

        if not dry_run:
            conn.commit()

        total += len(results)

        if total % 10000 < batch_size:
            elapsed = time.time() - t0
            print(
                f"  {total:>8,} points | {doc_count:>6,} docs | {chunk_count:>8,} chunks"
                f" | {sent_count:>4,} sent | {emp_count:>4,} emp | {elapsed:.0f}s",
                flush=True,
            )

        if next_offset is None:
            break
        offset = next_offset

    if not dry_run:
        print("\nMigrating ingest run history...")
        _migrate_ingest_runs(cur)
        conn.commit()
        cur.close()
        conn.close()

    elapsed = time.time() - t0
    print(f"\n{'Dry run complete' if dry_run else 'Migration complete'}")
    print(f"  Points scanned:   {total:,}")
    print(f"  Documents:        {doc_count:,}")
    print(f"  Chunks:           {chunk_count:,}")
    print(f"  Citations:        {cite_count:,}")
    print(f"  Sentencing rows:  {sent_count:,}")
    print(f"  Employment rows:  {emp_count:,}")
    print(f"  Elapsed:          {elapsed:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Qdrant data to PostgreSQL.")
    parser.add_argument("--dry-run",  action="store_true", help="Scan without writing")
    parser.add_argument("--court",    help="Limit to a single court code")
    parser.add_argument("--no-text",  action="store_true", help="Skip storing chunk text")
    args = parser.parse_args()
    run(dry_run=args.dry_run, court_filter=args.court, store_text=not args.no_text)


if __name__ == "__main__":
    main()
