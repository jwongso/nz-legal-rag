"""Backfill 'sentencing' payload on criminal court chunks in Qdrant.

Scans NZHC, NZCA, NZSC chunks only (criminal courts). Creates payload indexes
for filtered search, then writes sentencing data to chunks where found.

Usage:
    python -m ingest.sentencing_pipeline              # full backfill
    python -m ingest.sentencing_pipeline --dry-run    # count without writing
    python -m ingest.sentencing_pipeline --batch-size 200
"""

import argparse
import json

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

import config
from ingest.sentencing import extract_sentencing

_CRIMINAL_COURTS = ["NZHC", "NZCA", "NZSC"]


def _create_indexes(client: QdrantClient) -> None:
    specs = [
        ("sentencing.has_data",               "bool"),
        ("sentencing.starting_point_months",  "float"),
        ("sentencing.final_sentence_months",  "float"),
        ("sentencing.home_detention_months",  "float"),
        ("sentencing.guilty_plea_discount_pct", "float"),
        ("sentencing.sentence_type",          "keyword"),
        ("sentencing.has_guilty_plea",        "bool"),
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
        must=[FieldCondition(key="court", match=MatchAny(any=_CRIMINAL_COURTS))]
    )

    total = 0
    with_data = 0
    sentence_types: dict[str, int] = {}
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
            sentencing = extract_sentencing(text)

            key = json.dumps(sentencing, sort_keys=True)
            if key not in groups:
                groups[key] = {"payload": {"sentencing": sentencing}, "ids": []}
            groups[key]["ids"].append(point.id)

            if sentencing.get("has_data"):
                with_data += 1
                st = sentencing.get("sentence_type", "unknown")
                sentence_types[st] = sentence_types.get(st, 0) + 1

        if not dry_run:
            for group in groups.values():
                client.set_payload(
                    collection_name=config.QDRANT_COLLECTION,
                    payload=group["payload"],
                    points=group["ids"],
                )

        total += len(points)
        if total % 2000 == 0 or next_offset is None:
            print(f"  {total:,} processed | {with_data:,} with sentencing data...")

        if next_offset is None:
            break
        offset = next_offset

    prefix = "[DRY RUN] " if dry_run else ""
    pct = 100 * with_data / max(total, 1)
    print(f"\n{prefix}Done.")
    print(f"  Total criminal chunks:  {total:,}")
    print(f"  With sentencing data:   {with_data:,} ({pct:.1f}%)")

    if sentence_types:
        print("\nSentence type breakdown:")
        for st, count in sorted(sentence_types.items(), key=lambda x: -x[1]):
            print(f"  {st:<25} {count:>6,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill sentencing data on criminal chunks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    run_backfill(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
