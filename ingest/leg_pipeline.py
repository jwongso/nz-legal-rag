"""
Ingest NZ Acts from legislation.govt.nz into Qdrant.

Each section becomes one or more chunks using the same 120-word sliding window
as court decisions. case_id is per-section so dedup works correctly at retrieval.

Usage:
    python -m ingest.leg_pipeline --acts RTA ERA2000 PA2020
    python -m ingest.leg_pipeline  # defaults to all acts
"""

import argparse
import asyncio
import os

import config
from ingest.chunker import Chunk
from ingest.legislation import ACTS, LegSection, scrape_act
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


def _word_count(text: str) -> int:
    return len(text.split())


def _split_by_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + max_words]))
        i += max_words - overlap_words
    return chunks


def chunk_section(section: LegSection) -> list[Chunk]:
    """Convert a LegSection into Chunk objects using the same window size as court decisions."""
    s_ref = f"s{section.section_num}" if section.section_num else ""
    heading = f"{s_ref} {section.section_title}".strip()

    # case_id is per-section so RAG dedup allows multiple sections in a result set
    case_id = (
        f"NZLEG/{section.act_code}/s{section.section_num}"
        if section.section_num
        else f"NZLEG/{section.act_code}/{section.dlm_id}"
    )

    if _word_count(section.text) <= config.CHUNK_SIZE:
        full_text = f"{heading}\n\n{section.text}".strip() if heading else section.text
        if _word_count(full_text) < config.CHUNK_MIN_WORDS:
            return []
        return [Chunk(
            chunk_id=f"{case_id}#0",
            case_id=case_id,
            court="NZLEG",
            court_name=section.act_title,
            year=section.act_year,
            title=heading,
            date=str(section.act_year),
            parties=[],
            url=section.url,
            text=full_text,
            section_heading=heading,
            chunk_index=0,
            citations=[],
        )]

    sub_texts = _split_by_words(section.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    chunks: list[Chunk] = []
    for i, sub in enumerate(sub_texts):
        if _word_count(sub) < config.CHUNK_MIN_WORDS:
            continue
        text = f"{heading}\n\n{sub}".strip() if heading else sub
        chunks.append(Chunk(
            chunk_id=f"{case_id}#{i}",
            case_id=case_id,
            court="NZLEG",
            court_name=section.act_title,
            year=section.act_year,
            title=heading,
            date=str(section.act_year),
            parties=[],
            url=section.url,
            text=text,
            section_heading=heading,
            chunk_index=i,
            citations=[],
        ))
    return chunks


async def run(act_codes: list[str]) -> None:
    embedder = Embedder()
    store = VectorStore()
    store.ensure_collection()

    grand_total = 0
    for act_code in act_codes:
        print(f"\nIngesting {ACTS[act_code]['title']} ...")
        sections = await scrape_act(act_code)
        if not sections:
            continue

        act_chunks = 0
        for section in sections:
            chunks = chunk_section(section)
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

            try:
                store.upsert(vectors, payloads)
                act_chunks += len(chunks)
            except Exception as e:
                print(f"  [s{section.section_num}] SKIP - upsert failed: {e}")

        print(f"  {act_code}: {act_chunks} chunks indexed.")
        grand_total += act_chunks

    await embedder.close()
    print(f"\nDone. Total: {grand_total} legislation chunks indexed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NZ legislation into Qdrant")
    parser.add_argument(
        "--acts",
        nargs="+",
        choices=[*ACTS.keys(), "all"],
        default=["all"],
        metavar="ACT",
        help=f"Act codes to ingest. Choices: {', '.join(ACTS)} (default: all)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="CPU threads for embedding (default: 16)",
    )
    args = parser.parse_args()

    act_codes = list(ACTS.keys()) if "all" in args.acts else args.acts
    _limit_threads(args.threads)
    asyncio.run(run(act_codes))


if __name__ == "__main__":
    main()
