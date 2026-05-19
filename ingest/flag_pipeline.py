"""Single-pass backfill of 'flags' and 'penalty' payloads on all Qdrant chunks.

Runs detect_flags() and extract_penalty() in one scroll pass so every chunk
is read only once. Creates payload indexes for filtered search before writing.

Usage:
    python -m ingest.flag_pipeline              # run full backfill
    python -m ingest.flag_pipeline --dry-run    # count/check without writing
    python -m ingest.flag_pipeline --batch-size 200
"""

import argparse
import json

from qdrant_client import QdrantClient

import config
from ingest.flags import FLAG_LABELS, detect_flags
from ingest.penalty import extract_penalty


def _create_indexes(client: QdrantClient) -> None:
    specs = [
        ("flags",                  "keyword"),
        ("penalty.has_data",       "bool"),
        ("penalty.outcome_osi",    "float"),
        ("penalty.recovery_rate",  "float"),
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

    total = 0
    flagged_chunks = 0
    penalty_chunks = 0
    flag_counts: dict[str, int] = {}
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        # Group by (flags, penalty) combination for batched set_payload.
        # Most chunks share common penalty dicts (has_data=False, same court_type),
        # so this cuts write calls dramatically vs one call per point.
        groups: dict[str, dict] = {}

        for point in points:
            text  = point.payload.get("text", "")
            court = point.payload.get("court", "NZHC")
            title = point.payload.get("title", "")

            flags   = detect_flags(f"{title} {text}")
            penalty = extract_penalty(court, text)

            key = json.dumps({"f": flags, "p": penalty}, sort_keys=True)
            if key not in groups:
                groups[key] = {"payload": {"flags": flags, "penalty": penalty}, "ids": []}
            groups[key]["ids"].append(point.id)

            if flags:
                flagged_chunks += 1
                for f in flags:
                    flag_counts[f] = flag_counts.get(f, 0) + 1
            if penalty.get("has_data"):
                penalty_chunks += 1

        if not dry_run:
            for group in groups.values():
                client.set_payload(
                    collection_name=config.QDRANT_COLLECTION,
                    payload=group["payload"],
                    points=group["ids"],
                )

        total += len(points)
        if total % 5000 == 0 or next_offset is None:
            print(
                f"  {total:,} processed | "
                f"{flagged_chunks:,} flagged | "
                f"{penalty_chunks:,} with penalty data..."
            )

        if next_offset is None:
            break
        offset = next_offset

    prefix = "[DRY RUN] " if dry_run else ""
    pct_f = 100 * flagged_chunks / max(total, 1)
    pct_p = 100 * penalty_chunks / max(total, 1)

    print(f"\n{prefix}Done.")
    print(f"  Total chunks:       {total:,}")
    print(f"  Flagged chunks:     {flagged_chunks:,} ({pct_f:.1f}%)")
    print(f"  With penalty data:  {penalty_chunks:,} ({pct_p:.1f}%)")

    if flag_counts:
        print("\nFlag breakdown:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            label = FLAG_LABELS.get(flag, flag)
            print(f"  {label:<45} {count:>6,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill flags and penalty data on all Qdrant chunks"
    )
    parser.add_argument("--dry-run", action="store_true", help="Count without writing")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    run_backfill(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
