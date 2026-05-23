"""Secondary source ingestion pipeline (Phase 1 + 2).

Reads files from data/inbox/, parses them, chunks them, embeds into the
nz_legal_secondary Qdrant collection, records everything in PostgreSQL, then
runs citation extraction to link secondary chunks to primary corpus documents.

Files are deduplicated by SHA-256 hash - dropping the same PDF twice is a no-op.
Processed files are moved to data/processed/; failed files to data/failed/.

Supported formats: .pdf .docx .doc .txt .md

Run:
    python -m ingest.ingest_secondary
    python -m ingest.ingest_secondary --inbox data/inbox --source-type journal_article
    python -m ingest.ingest_secondary --file path/to/paper.pdf --source-type legal_memo
"""

import argparse
import hashlib
import shutil
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from ingest.secondary_chunker import chunk_secondary
from ingest.secondary_citations import process_document as extract_citations
from ingest.secondary_parser import parse
from rag.embedder import Embedder

_SECONDARY_COLLECTION = "nz_legal_secondary"
_SUPPORTED = {".pdf", ".docx", ".doc", ".txt", ".md"}

_INBOX    = Path("data/inbox")
_DONE     = Path("data/processed")
_FAILED   = Path("data/failed")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _ensure_collection(client: QdrantClient, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if _SECONDARY_COLLECTION not in existing:
        client.create_collection(
            collection_name=_SECONDARY_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field, schema in [("source_type", "keyword"), ("doc_id", "keyword"),
                               ("chunk_type", "keyword"), ("year", "integer")]:
            client.create_payload_index(
                collection_name=_SECONDARY_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        print(f"  Created Qdrant collection '{_SECONDARY_COLLECTION}' dim={dim}")
    else:
        print(f"  Qdrant collection '{_SECONDARY_COLLECTION}' already exists")


def _already_ingested(conn, file_hash: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT id FROM secondary_documents WHERE file_hash = %s", (file_hash,))
    return cur.fetchone() is not None


def _insert_doc(conn, doc_id: str, source_type: str, parsed, file_path: str, file_hash: str) -> None:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO secondary_documents
            (id, source_type, title, authors, publication_year,
             file_path, file_hash, parse_status, parse_method)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'chunked', %s)
    """, (
        doc_id, source_type,
        parsed.title, parsed.authors or [],
        parsed.publication_year,
        file_path, file_hash,
        parsed.parse_method,
    ))
    conn.commit()


def _insert_chunks(conn, chunks) -> None:
    cur = conn.cursor()
    for c in chunks:
        cur.execute("""
            INSERT INTO secondary_chunks
                (document_id, chunk_index, section_title, chunk_type, text, token_count)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (c.doc_id, c.chunk_index, c.section_title, c.chunk_type, c.text, c.token_count))
    conn.commit()


def _update_chunk_qdrant_ids(conn, doc_id: str, chunk_index_to_point_id: dict) -> None:
    cur = conn.cursor()
    for chunk_index, point_id in chunk_index_to_point_id.items():
        cur.execute("""
            UPDATE secondary_chunks SET qdrant_point_id = %s
            WHERE document_id = %s AND chunk_index = %s
        """, (point_id, doc_id, chunk_index))
    cur.execute("""
        UPDATE secondary_documents SET parse_status = 'embedded', updated_at = now()
        WHERE id = %s
    """, (doc_id,))
    conn.commit()


def _mark_failed(conn, doc_id: str, error: str) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE secondary_documents
        SET parse_status = 'failed', parse_error = %s, updated_at = now()
        WHERE id = %s
    """, (error[:1000], doc_id))
    conn.commit()


def ingest_file(
    path: Path,
    source_type: str,
    conn,
    client: QdrantClient,
    embedder: Embedder,
) -> bool:
    file_hash = _sha256(path)

    if _already_ingested(conn, file_hash):
        print(f"  [skip] {path.name} already ingested (hash match)")
        return False

    print(f"  Parsing {path.name} ...")
    try:
        parsed = parse(path)
    except Exception as e:
        print(f"  [error] parse failed: {e}")
        return False

    doc_id = str(uuid.uuid4())
    _insert_doc(conn, doc_id, source_type, parsed, str(path), file_hash)

    chunks = chunk_secondary(doc_id, parsed.text)
    if not chunks:
        print(f"  [warn] no chunks produced for {path.name}")
        return False

    _insert_chunks(conn, chunks)
    print(f"  {len(chunks)} chunks  ({parsed.parse_method})")

    # Embed and upsert to Qdrant
    texts = [c.text for c in chunks]
    vecs = embedder.encode_documents(texts)

    points = []
    idx_to_pid: dict[int, str] = {}
    for c, vec in zip(chunks, vecs):
        point_id = str(uuid.uuid4())
        idx_to_pid[c.chunk_index] = point_id
        points.append(PointStruct(
            id=point_id,
            vector=vec,
            payload={
                "doc_id":       c.doc_id,
                "chunk_index":  c.chunk_index,
                "source_type":  source_type,
                "chunk_type":   c.chunk_type,
                "section_title": c.section_title,
                "title":        parsed.title,
                "authors":      parsed.authors,
                "year":         parsed.publication_year or 0,
                "text":         c.text,
            },
        ))

    client.upsert(collection_name=_SECONDARY_COLLECTION, points=points)
    _update_chunk_qdrant_ids(conn, doc_id, idx_to_pid)
    print(f"  Upserted {len(points)} vectors to '{_SECONDARY_COLLECTION}'")

    # Phase 2: citation extraction
    cit = extract_citations(conn, doc_id)
    print(f"  Citations: {cit['total_citations']} found, "
          f"{cit['linked_to_corpus']} linked to primary corpus")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Secondary source ingestion")
    parser.add_argument("--inbox", default=str(_INBOX),
                        help="Folder to scan for new files (default: data/inbox)")
    parser.add_argument("--file", default=None,
                        help="Ingest a single specific file instead of scanning inbox")
    parser.add_argument("--source-type", default="journal_article",
                        choices=["journal_article", "law_review", "legal_memo",
                                 "commentary", "user_note"],
                        help="Source type tag (default: journal_article)")
    parser.add_argument("--no-move", action="store_true",
                        help="Do not move files to processed/failed after ingest")
    parser.add_argument("--qdrant-url", default=config.QDRANT_URL)
    args = parser.parse_args()

    conn   = psycopg2.connect(dbname="nz_legal")
    client = QdrantClient(url=args.qdrant_url)
    embedder = Embedder()  # uses production embed model (nomic)

    _ensure_collection(client, embedder.dim)

    _DONE.mkdir(parents=True, exist_ok=True)
    _FAILED.mkdir(parents=True, exist_ok=True)

    if args.file:
        files = [Path(args.file)]
    else:
        inbox = Path(args.inbox)
        inbox.mkdir(parents=True, exist_ok=True)
        files = [f for f in inbox.iterdir()
                 if f.is_file() and f.suffix.lower() in _SUPPORTED]
        if not files:
            print(f"No supported files found in {inbox}")
            return

    print(f"Secondary source ingest")
    print(f"  source_type: {args.source_type}")
    print(f"  files:       {len(files)}")
    print()

    ok = 0
    for f in sorted(files):
        print(f"[{f.name}]")
        success = ingest_file(f, args.source_type, conn, client, embedder)
        if not args.no_move and args.file is None:
            dest = (_DONE if success else _FAILED) / f.name
            shutil.move(str(f), dest)
            print(f"  -> {dest}")
        if success:
            ok += 1
        print()

    conn.close()
    print(f"Done. {ok}/{len(files)} files ingested.")


if __name__ == "__main__":
    main()
