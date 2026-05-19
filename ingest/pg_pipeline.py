"""Backfill 'pg' (personal grievance outcome) payload on ERA/NZEmpC chunks.

Scans NZERA and NZEmpC chunks only. Creates payload indexes for filtered
search, then writes PG outcome data to chunks where found.

Usage:
    python -m ingest.pg_pipeline              # full backfill
    python -m ingest.pg_pipeline --dry-run    # count without writing
    python -m ingest.pg_pipeline --batch-size 200
"""

import argparse
import json

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

import config
from ingest.pg_outcome import extract_pg_outcome

_PG_COURTS = ["NZERA", "NZEmpC"]


def _create_indexes(client: QdrantClient) -> None:
    specs = [
        ("pg.has_data",                "bool"),
        ("pg.reinstatement_ordered",   "bool"),
        ("pg.contributory_conduct_pct","float"),
        ("pg.grievance_types",         "keyword"),
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


def run_backfill(dry_run: bool = False, batch_size: int = 200) -> None:
    client = QdrantClient(url=config.QDRANT_URL)

    if not dry_run:
        print("Creating payload indexes...")
        _create_indexes(client)

    query_filter = Filter(
        must=[FieldCondition(key="court", match=MatchAny(any=_PG_COURTS))]
    )

    total = 0
    with_data = 0
    type_counts: dict[str, int] = {}
    reinstated = 0
    declined = 0
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=query_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        groups: dict[str, dict] = {}
        for point in points:
            text = point.payload.get("text", "")
            pg = extract_pg_outcome(text)

            key = json.dumps(pg, sort_keys=True)
            if key not in groups:
                groups[key] = {"payload": {"pg": pg}, "ids": []}
            groups[key]["ids"].append(point.id)

            if pg.get("has_data"):
                with_data += 1
                for gt in pg.get("grievance_types", []):
                    type_counts[gt] = type_counts.get(gt, 0) + 1
                if pg.get("reinstatement_ordered") is True:
                    reinstated += 1
                elif pg.get("reinstatement_ordered") is False:
                    declined += 1

        if not dry_run:
            for group in groups.values():
                client.set_payload(
                    collection_name=config.QDRANT_COLLECTION,
                    payload=group["payload"],
                    points=group["ids"],
                )

        total += len(points)
        if total % 2000 == 0 or next_offset is None:
            print(f"  {total:,} processed | {with_data:,} with PG data...")

        if next_offset is None:
            break
        offset = next_offset

    prefix = "[DRY RUN] " if dry_run else ""
    pct = 100 * with_data / max(total, 1)
    print(f"\n{prefix}Done.")
    print(f"  Total ERA/NZEmpC chunks: {total:,}")
    print(f"  With PG data:            {with_data:,} ({pct:.1f}%)")
    print(f"  Reinstatement ordered:   {reinstated:,}")
    print(f"  Reinstatement declined:  {declined:,}")

    if type_counts:
        print("\nGrievance type breakdown:")
        for gt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {gt:<35} {count:>6,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill PG outcome data on ERA/NZEmpC chunks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    run_backfill(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
