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
import time

import config
from ingest.chunker import chunk_case
from ingest.moj_scraper import scrape_moj
from rag.embedder import Embedder
from rag.retriever import VectorStore


async def run(dry_run: bool = False, limit: int | None = None) -> None:
    embedder = Embedder()
    store = VectorStore(collection=config.QDRANT_TENANCY_COLLECTION)
    if not dry_run:
        store.ensure_collection()

    total_docs = 0
    total_chunks = 0
    t0 = time.time()

    async for doc in scrape_moj(verbose=True):
        if limit and total_docs >= limit:
            break

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
        vectors = await embedder.embed_batch(texts)

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

        if total_docs % 100 == 0:
            elapsed = time.time() - t0
            rate = total_docs / elapsed
            print(f"  {total_docs} docs, {total_chunks} chunks | {rate:.1f} docs/s")

    elapsed = time.time() - t0
    print(f"\nFinished: {total_docs} docs, {total_chunks} chunks in {elapsed:.0f}s")
    if dry_run:
        print("(dry run - nothing written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MoJ TT decisions into Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no writes")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N documents")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    main()
