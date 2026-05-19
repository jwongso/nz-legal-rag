"""Backfill 'counsel' payload on all Qdrant chunks.

Phase 1: Scan chunks 0-3 for every case to extract the appearances block
         and build a {case_id -> counsel dict} map.

Phase 2: Full scan of all 182k chunks. For cases that have counsel data,
         accumulate all point IDs per case, then write counsel to every
         chunk of that case in one set_payload call per case.

Writing counsel to every chunk (not just early ones) means scroll_notable
always surfaces counsel data on whichever chunk it returns for a case.

Usage:
    python -m ingest.counsel_pipeline
    python -m ingest.counsel_pipeline --dry-run
    python -m ingest.counsel_pipeline --batch-size 500
"""

import argparse
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, Range

import config
from ingest.counsel import extract_counsel

_EARLY_CHUNKS = 4  # scan chunk_index 0..3


def _create_indexes(client: QdrantClient) -> None:
    specs = [
        ("counsel.has_data",    "bool"),
        ("counsel.all_surnames","keyword"),
        ("counsel.crown",       "keyword"),
        ("counsel.defence",     "keyword"),
    ]
    for field, schema in specs:
        try:
            client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
            print(f"  Created index: {field} ({schema})")
        except Exception:
            print(f"  Index already exists: {field}")


def run_backfill(dry_run: bool = False, batch_size: int = 500) -> None:
    client = QdrantClient(url=config.QDRANT_URL)

    if not dry_run:
        print("Creating payload indexes...")
        _create_indexes(client)

    # ------------------------------------------------------------------
    # Phase 1: Extract counsel from early chunks (chunk_index 0-3)
    # ------------------------------------------------------------------
    print(f"\nPhase 1: extracting counsel from first {_EARLY_CHUNKS} chunks per case...")

    case_counsel: dict[str, dict] = {}   # case_id -> counsel dict
    early_total = 0
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(
                    key="chunk_index",
                    range=Range(gte=0, lte=_EARLY_CHUNKS - 1),
                )
            ]),
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            case_id = p.payload.get("case_id", "")
            if not case_id or case_id in case_counsel:
                continue  # already found for this case
            text = p.payload.get("text", "")
            result = extract_counsel(text)
            if result["has_data"]:
                case_counsel[case_id] = result

        early_total += len(points)
        if next_offset is None:
            print(f"  Scanned {early_total:,} early chunks -> "
                  f"counsel found for {len(case_counsel):,} cases")
            break
        offset = next_offset

    if not case_counsel:
        print("No counsel data found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Phase 2: Full scan - accumulate all point IDs per case
    # ------------------------------------------------------------------
    print("\nPhase 2: scanning all chunks to collect point IDs per case...")

    case_point_ids: dict[str, list] = defaultdict(list)
    total = 0
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=["case_id"],  # only need case_id - faster
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            case_id = p.payload.get("case_id", "")
            if case_id in case_counsel:
                case_point_ids[case_id].append(p.id)

        total += len(points)
        if total % 20000 == 0 or next_offset is None:
            print(f"  {total:,} chunks scanned...")

        if next_offset is None:
            break
        offset = next_offset

    # ------------------------------------------------------------------
    # Write counsel to all chunks of each matched case
    # ------------------------------------------------------------------
    print(f"\nWriting counsel to {len(case_point_ids):,} cases "
          f"({sum(len(v) for v in case_point_ids.values()):,} chunks)...")

    written_cases = 0
    surname_counts: dict[str, int] = {}
    crown_counts:   dict[str, int] = {}

    for case_id, point_ids in case_point_ids.items():
        counsel = case_counsel[case_id]

        if not dry_run:
            # One set_payload call covers all chunks of this case
            client.set_payload(
                collection_name=config.QDRANT_COLLECTION,
                payload={"counsel": counsel},
                points=point_ids,
            )

        written_cases += 1
        for s in counsel.get("all_surnames", []):
            surname_counts[s] = surname_counts.get(s, 0) + 1
        for c in counsel.get("crown", []):
            crown_counts[c] = crown_counts.get(c, 0) + 1

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Done.")
    print(f"  Cases with counsel written: {written_cases:,}")
    print(f"  Unique surnames indexed:    {len(surname_counts):,}")
    print(f"  Unique crown counsel:       {len(crown_counts):,}")

    if surname_counts:
        print("\nTop 20 most frequent surnames:")
        for name, count in sorted(surname_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {name:<30} {count:>4,} cases")

    if crown_counts:
        print("\nTop 15 crown counsel (by cases):")
        for name, count in sorted(crown_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {name:<30} {count:>4,} cases")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill counsel/appearances data on all Qdrant chunks"
    )
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    run_backfill(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
