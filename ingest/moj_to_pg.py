#!/usr/bin/env python3
"""
Migrate nztt_moj Qdrant collection into PostgreSQL documents + chunks tables.
Idempotent - uses ON CONFLICT DO NOTHING, safe to re-run.

Usage:
    python -m ingest.moj_to_pg
"""

import sys
from datetime import datetime

import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient

COLLECTION = "nztt_moj"
SCROLL_BATCH = 1000
INSERT_BATCH = 500


def _parse_date(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


def _run():
    qclient = QdrantClient(url="http://localhost:6333")
    conn = psycopg2.connect(dbname="nz_legal")
    conn.autocommit = False
    cur = conn.cursor()

    # case_id -> pg document id (built incrementally)
    case_doc_id: dict[str, int] = {}

    # Pre-load any existing MoJ documents (re-run safety)
    cur.execute(
        "SELECT id, citation FROM documents WHERE court = 'NZTT' AND citation LIKE 'NZTT-MOJ-%'"
    )
    for doc_id, citation in cur.fetchall():
        case_doc_id[citation] = doc_id
    print(f"Pre-loaded {len(case_doc_id)} existing MoJ documents from PostgreSQL")

    offset = None
    total_docs = 0
    total_chunks = 0
    skipped_chunks = 0

    while True:
        points, next_offset = qclient.scroll(
            COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        # --- Phase 1: collect new documents in this batch ---
        new_cases: dict[str, dict] = {}
        for p in points:
            pl = p.payload
            cid = pl.get("case_id", "")
            if not cid or cid in case_doc_id or cid in new_cases:
                continue
            new_cases[cid] = {
                "citation": cid,  # use case_id as citation key for MoJ docs
                "court": "NZTT",
                "title": (pl.get("title") or "")[:500],
                "decision_date": _parse_date(pl.get("date")),
                "source_url": pl.get("url") or "",
                "jurisdiction": "NZ",
                "ingestion_status": "complete",
            }

        if new_cases:
            rows = list(new_cases.values())
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO documents
                    (citation, court, title, decision_date, source_url, jurisdiction, ingestion_status)
                VALUES %s
                ON CONFLICT (citation) DO NOTHING
                """,
                [
                    (
                        r["citation"], r["court"], r["title"],
                        r["decision_date"], r["source_url"],
                        r["jurisdiction"], r["ingestion_status"],
                    )
                    for r in rows
                ],
            )
            conn.commit()

            # Fetch IDs for all new cases (including those that already existed)
            cids = list(new_cases.keys())
            cur.execute(
                "SELECT id, citation FROM documents WHERE citation = ANY(%s)",
                (cids,),
            )
            for doc_id, citation in cur.fetchall():
                case_doc_id[citation] = doc_id
            total_docs += len(new_cases)

        # --- Phase 2: insert chunks ---
        chunk_rows = []
        for p in points:
            pl = p.payload
            cid = pl.get("case_id", "")
            doc_id = case_doc_id.get(cid)
            if doc_id is None:
                skipped_chunks += 1
                continue
            chunk_rows.append((
                doc_id,
                pl.get("chunk_index", 0),
                pl.get("text") or "",
                str(p.id),
                (pl.get("section_heading") or "")[:500],
            ))

        if chunk_rows:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks (document_id, chunk_index, text, qdrant_point_id, section_title)
                VALUES %s
                ON CONFLICT (qdrant_point_id) DO NOTHING
                """,
                chunk_rows,
                page_size=INSERT_BATCH,
            )
            conn.commit()
            total_chunks += len(chunk_rows)

        print(
            f"  docs={total_docs}  chunks={total_chunks}  skipped={skipped_chunks}",
            end="\r",
            flush=True,
        )

        if next_offset is None:
            break
        offset = next_offset

    cur.close()
    conn.close()
    print(f"\nDone.  {total_docs} documents,  {total_chunks} chunks inserted.")


if __name__ == "__main__":
    _run()
