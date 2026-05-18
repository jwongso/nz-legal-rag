"""
Orchestrates scrape -> chunk -> embed -> upsert into Qdrant.

Current hardware: single machine, llama-server on host GPU.
# TODO: Once AI Max+ 395 Node 2 is available, run LoRA fine-tuning here on the
#       ingested corpus to domain-adapt Qwen3-8B to NZ legal language. Use
#       unsloth or axolotl with the JSONL output from data/finetune/.
#
# TODO: Once AI Max+ 395 Node 3 is available, move Qdrant and Ollama there
#       (persistent storage node) and point QDRANT_URL / OLLAMA_URL env vars
#       at Node 3's IP. This frees Node 1 for pure inference.

Usage:
    python -m ingest.pipeline --court NZTT --years 2022 2023 2024
    python -m ingest.pipeline --court NZHC --years 2023 --max-per-year 100 --threads 16
"""

import argparse
import asyncio
import os

from ingest.chunker import chunk_case
from ingest.scraper import scrape_court
from rag.embedder import Embedder
from rag.retriever import VectorStore


def _limit_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


async def run(court: str, years: list[int], max_per_year: int) -> None:
    embedder = Embedder()
    store = VectorStore()
    store.ensure_collection()

    total_chunks = 0
    async for doc in scrape_court(court, years, max_per_year=max_per_year):
        chunks = chunk_case(doc)
        if not chunks:
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
        total_chunks += len(chunks)
        print(f"  [{doc.case_id}] {doc.title[:60]} -> {len(chunks)} chunks")

    print(f"\nDone. Indexed {total_chunks} chunks from {court} {years}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--court", required=True, choices=["NZTT", "NZHC", "NZCA", "NZSC", "NZEmpC", "NZERA"])
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--max-per-year", type=int, default=200)
    parser.add_argument("--threads", type=int, default=16, help="CPU threads for embedding (default: 16)")
    args = parser.parse_args()
    _limit_threads(args.threads)
    asyncio.run(run(args.court, args.years, args.max_per_year))


if __name__ == "__main__":
    main()
