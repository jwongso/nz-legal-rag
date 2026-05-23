"""Partial re-ingest for the embedding model shootout.

Creates a new Qdrant collection for a candidate embedding model using a
benchmark subset of chunks. The production nz_legal collection is never
modified.

Strategy:
  1. Gold pass  - fetch ALL chunks for expected + acceptable documents from
                  the retrieval gold set (these must be present for H@5 to work)
  2. Fill pass  - for each court, add up to --per-court-limit additional chunks
                  sampled in document order (oldest documents first)

Total ingest is typically 100-130k chunks depending on per-court-limit.

Run:
    python -m benchmarks.runners.run_embed_ingest --model BAAI/bge-m3 --collection nz_legal_bge_m3
    python -m benchmarks.runners.run_embed_ingest --model intfloat/e5-large-v2 --collection nz_legal_e5
    python -m benchmarks.runners.run_embed_ingest --model Qwen/Qwen3-Embedding-0.6B --collection nz_legal_qwen3_06b
"""

import argparse
import json
import time
from pathlib import Path

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from rag.embedder import Embedder

_GOLD_PATH = Path("benchmarks/datasets/retrieval_gold.jsonl")
_DEFAULT_COURTS = ["NZERA", "NZEmpC", "NZCA", "NZTT", "NZHC", "NZLEG", "NZHRRT", "NZACC"]
_DEFAULT_PER_COURT_LIMIT = 15000
_UPSERT_BATCH = 128


def _get_gold_doc_ids(gold_path: Path) -> set[str]:
    ids: set[str] = set()
    for line in gold_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ids.update(r.get("expected_documents", []))
        ids.update(r.get("acceptable_documents", []))
    return ids


def _fetch_chunks(
    conn,
    courts: list[str],
    per_court_limit: int,
    gold_ids: set[str],
) -> list[dict]:
    cur = conn.cursor()
    chunks: list[dict] = []
    seen_pids: set[str] = set()

    def _row_to_chunk(row) -> dict:
        pid, text, citation, chunk_idx, court, title, url, year = row
        return {
            "point_id": pid,
            "text": text or "",
            "case_id": citation,
            "chunk_index": int(chunk_idx),
            "court": court,
            "title": title or "",
            "url": url or "",
            "year": int(year) if year else 0,
        }

    # Gold pass: all chunks for expected/acceptable documents (within target courts)
    if gold_ids:
        cur.execute("""
            SELECT c.qdrant_point_id, c.text, d.citation, c.chunk_index,
                   d.court, d.title, d.source_url,
                   EXTRACT(YEAR FROM d.decision_date)::int
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.citation = ANY(%s)
              AND d.court = ANY(%s)
              AND c.qdrant_point_id IS NOT NULL
        """, (list(gold_ids), courts))
        for row in cur.fetchall():
            pid = row[0]
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            chunks.append(_row_to_chunk(row))
        print(f"  Gold pass: {len(chunks)} chunks from {len(gold_ids)} gold documents")

    # Fill pass: sample each court up to per_court_limit additional chunks
    for court in courts:
        cur.execute("""
            SELECT c.qdrant_point_id, c.text, d.citation, c.chunk_index,
                   d.court, d.title, d.source_url,
                   EXTRACT(YEAR FROM d.decision_date)::int
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.court = %s
              AND c.qdrant_point_id IS NOT NULL
            ORDER BY d.id, c.chunk_index
            LIMIT %s
        """, (court, per_court_limit))
        added = 0
        for row in cur.fetchall():
            pid = row[0]
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            chunks.append(_row_to_chunk(row))
            added += 1
        print(f"  Fill pass {court}: +{added} chunks")

    return chunks


def _ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        print(f"  Dropping existing collection '{name}'")
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    for field, schema in [("court", "keyword"), ("year", "integer"), ("case_id", "keyword")]:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema,
        )
    print(f"  Created collection '{name}' dim={dim}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Partial re-ingest for embedding benchmark")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID (e.g. BAAI/bge-m3)")
    parser.add_argument("--collection", required=True,
                        help="Target Qdrant collection name")
    parser.add_argument("--courts", default=",".join(_DEFAULT_COURTS),
                        help="Comma-separated court codes (default: all gold courts)")
    parser.add_argument("--per-court-limit", type=int, default=_DEFAULT_PER_COURT_LIMIT,
                        help="Max fill chunks per court (default: 15000)")
    parser.add_argument("--gold", default=str(_GOLD_PATH),
                        help="Retrieval gold JSONL (gold docs always included)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size (default: 64)")
    parser.add_argument("--device", default=None,
                        help="Torch device: cpu or cuda (default: cuda if available, else cpu)")
    parser.add_argument("--qdrant-url", default=config.QDRANT_URL)
    args = parser.parse_args()

    courts = [c.strip() for c in args.courts.split(",") if c.strip()]
    gold_ids = _get_gold_doc_ids(Path(args.gold))

    print(f"Embedding ingest")
    print(f"  model:            {args.model}")
    print(f"  collection:       {args.collection}")
    print(f"  courts:           {courts}")
    print(f"  per-court-limit:  {args.per_court_limit}")
    print(f"  gold documents:   {len(gold_ids)}")
    print()

    print("Loading embedder...")
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = Embedder(model_name=args.model, device=device)
    print(f"  dim={embedder.dim}  device={device}")
    print()

    print("Fetching chunks from PostgreSQL...")
    conn = psycopg2.connect(dbname="nz_legal")
    chunks = _fetch_chunks(conn, courts, args.per_court_limit, gold_ids)
    conn.close()
    print(f"  Total: {len(chunks):,} chunks to ingest")
    print()

    client = QdrantClient(url=args.qdrant_url)
    _ensure_collection(client, args.collection, embedder.dim)
    print()

    print(f"Embedding and upserting...")
    t_start = time.monotonic()
    n_done = 0
    report_every = max(1, len(chunks) // 20)

    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i: i + args.batch_size]
        texts = [c["text"] for c in batch]
        vecs = embedder.encode_documents(texts, batch_size=args.batch_size)

        points = [
            PointStruct(
                id=c["point_id"],
                vector=vec,
                payload={
                    "case_id": c["case_id"],
                    "chunk_index": c["chunk_index"],
                    "court": c["court"],
                    "title": c["title"],
                    "url": c["url"],
                    "year": c["year"],
                    "text": c["text"],
                },
            )
            for c, vec in zip(batch, vecs)
        ]
        client.upsert(collection_name=args.collection, points=points)
        n_done += len(batch)

        if n_done % report_every < args.batch_size or n_done >= len(chunks):
            elapsed = time.monotonic() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(chunks) - n_done) / rate / 60 if rate > 0 else 0
            print(f"  {n_done:,}/{len(chunks):,}  {rate:.0f} chunks/s  ETA {eta:.1f} min")

    total_s = time.monotonic() - t_start
    print()
    print(f"Done. {n_done:,} chunks in {total_s/60:.1f} min ({n_done/total_s:.0f} chunks/s avg)")
    print(f"  collection: {args.collection}  dim={embedder.dim}")


if __name__ == "__main__":
    main()
