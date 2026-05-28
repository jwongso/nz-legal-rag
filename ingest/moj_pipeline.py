"""
Ingest MoJ Tenancy Tribunal decisions into Qdrant.

Fetches from the MoJ public Solr index (no Playwright needed - direct HTTP).
Uses the same chunker and embedder as the main pipeline.

Usage:
    python -m ingest.moj_pipeline               # full ingest (~32k decisions)
    python -m ingest.moj_pipeline --dry-run     # count only, no writes
    python -m ingest.moj_pipeline --limit 500   # test with first 500 decisions
"""

import argparse
import asyncio
import os
import time

import config
from ingest.chunker import chunk_case
from ingest.moj_scraper import scrape_moj
from rag.embedder import Embedder
from rag.retriever import VectorStore


async def run(dry_run: bool = False, limit: int | None = None, resume_from: int = 0, nice: int = 0, since_date: str | None = None) -> None:
    if nice:
        os.nice(nice)
    embedder = Embedder()
    store = VectorStore(collection=config.QDRANT_TENANCY_COLLECTION)
    if not dry_run:
        store.ensure_collection()

    total_docs = 0
    skipped = 0
    total_chunks = 0
    t0 = time.time()

    # Buffer incoming docs so we can batch-check existence before embedding
    _CHECK_BATCH = 50
    buf: list = []

    async def _flush(docs: list) -> None:
        nonlocal total_docs, skipped, total_chunks
        if not docs:
            return
        # Batch existence check - one Qdrant round-trip per buffer
        case_ids = [d.case_id for d in docs]
        existing = store.case_ids_exist(case_ids) if not dry_run else set()
        new_docs = [d for d in docs if d.case_id not in existing]
        skipped += len(docs) - len(new_docs)

        for doc in new_docs:
            chunks = chunk_case(doc)
            if not chunks:
                continue
            total_docs += 1
            total_chunks += len(chunks)
            if dry_run:
                if total_docs <= 3:
                    print(f"  [{doc.case_id}] {doc.title} ({doc.date}) -> {len(chunks)} chunks")
                continue
            texts = [c.text for c in chunks]
            vectors = await embedder.embed_batch(texts, batch_size=32)
            payloads = [
                {
                    "chunk_id": c.chunk_id,
                    "case_id": c.case_id,
                    "court": c.court,
                    "court_name": c.court_name,
                    "year": c.year,
                    "title": c.title,
                    "date": c.date,
                    "parties": c.parties,
                    "url": c.url,
                    "text": c.text,
                    "section_heading": c.section_heading,
                    "chunk_index": c.chunk_index,
                    "citations": c.citations,
                }
                for c in chunks
            ]
            store.upsert(vectors, payloads)

        elapsed = time.time() - t0
        rate = total_docs / elapsed if elapsed > 0 else 0
        print(f"  {total_docs} new, {skipped} skipped | {rate:.1f} docs/s")

    async for doc in scrape_moj(verbose=True, resume_from=resume_from, since_date=since_date):
        if limit and (total_docs + len(buf)) >= limit:
            break
        buf.append(doc)
        if len(buf) >= _CHECK_BATCH:
            await _flush(buf)
            buf.clear()

    await _flush(buf)

    elapsed = time.time() - t0
    print(f"\nFinished: {total_docs} new docs, {skipped} skipped, {total_chunks} chunks in {elapsed:.0f}s")
    if dry_run:
        print("(dry run - nothing written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MoJ TT decisions into Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no writes")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N documents")
    parser.add_argument("--resume-from", type=int, default=0, help="Skip first N Solr records (resume after crash)")
    parser.add_argument("--nice", type=int, default=10, help="Process nice level 0-19 (default 10)")
    parser.add_argument("--since-date", default=None, help="Only ingest decisions on or after YYYY-MM-DD")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run, limit=args.limit, resume_from=args.resume_from, nice=args.nice, since_date=args.since_date))


if __name__ == "__main__":
    main()
