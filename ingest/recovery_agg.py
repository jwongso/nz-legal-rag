"""Case-level recovery rate aggregation for civil financial cases.

Chunk-level extraction in penalty.py can only compute recovery_rate when
both the claimed amount and the awarded amount appear in the same ~120-word
chunk. That is rare: claims are stated in the opening, awards in the orders.

This script fixes that by:
  1. Scrolling all civil-court chunks grouped by case_id
  2. Taking max(awarded_amount) and max(claimed_amount) across all chunks
  3. Writing recovery_rate + recovery_class back to the chunk that holds the
     max awarded_amount (the chunk scroll_notable will surface for that case)

Also creates a payload index on penalty.awarded_amount for filtered search.

Run this after flag_pipeline.py backfill completes.

Usage:
    python -m ingest.recovery_agg
    python -m ingest.recovery_agg --dry-run
"""

import argparse
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

import config
from ingest.penalty import _recovery_class

_CIVIL_COURTS = [
    "NZTT", "NZERA", "NZEmpC", "NZACC",
    "NZFC", "NZLCDT", "NZHRRT", "NZREADT",
]


def run_aggregation(dry_run: bool = False) -> None:
    client = QdrantClient(url=config.QDRANT_URL)

    if not dry_run:
        try:
            client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION,
                field_name="penalty.awarded_amount",
                field_schema="float",
            )
            print("Created index: penalty.awarded_amount (float)")
        except Exception:
            print("Index already exists: penalty.awarded_amount")

    # case_id -> aggregated civil data
    cases: dict[str, dict] = defaultdict(lambda: {
        "max_awarded": None,
        "max_awarded_id": None,
        "max_awarded_pen": None,  # full penalty dict of that chunk
        "max_claimed": None,
    })

    offset = None
    total_civil = 0

    print("Scanning civil chunks...")
    while True:
        points, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="court", match=MatchAny(any=_CIVIL_COURTS))
            ]),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            case_id = p.payload.get("case_id", "")
            pen = p.payload.get("penalty", {})
            awarded = pen.get("awarded_amount")
            claimed = pen.get("claimed_amount")
            c = cases[case_id]

            if awarded is not None:
                if c["max_awarded"] is None or awarded > c["max_awarded"]:
                    c["max_awarded"] = awarded
                    c["max_awarded_id"] = str(p.id)
                    c["max_awarded_pen"] = dict(pen)

            if claimed is not None:
                if c["max_claimed"] is None or claimed > c["max_claimed"]:
                    c["max_claimed"] = claimed

        total_civil += len(points)
        if total_civil % 10000 == 0 or next_offset is None:
            print(f"  {total_civil:,} civil chunks  |  {len(cases):,} cases...")

        if next_offset is None:
            break
        offset = next_offset

    # Compute recovery rates and write back
    updates = 0
    no_claim = 0
    sample: list[tuple] = []

    for case_id, c in cases.items():
        awarded = c["max_awarded"]
        claimed = c["max_claimed"]
        point_id = c["max_awarded_id"]

        if awarded is None or point_id is None:
            continue
        if claimed is None or claimed <= 0:
            no_claim += 1
            continue

        rate = round(awarded / claimed, 3)
        rc   = _recovery_class(rate)

        updated_pen = dict(c["max_awarded_pen"])
        updated_pen["recovery_rate"]  = rate
        updated_pen["recovery_class"] = rc
        # Persist the case-wide claimed_amount on this chunk too
        updated_pen["claimed_amount"] = claimed

        if not dry_run:
            client.set_payload(
                collection_name=config.QDRANT_COLLECTION,
                payload={"penalty": updated_pen},
                points=[point_id],
            )

        updates += 1
        if len(sample) < 15:
            sample.append((case_id, claimed, awarded, rate, rc))

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Done.")
    print(f"  Civil cases:             {len(cases):,}")
    print(f"  Recovery rate written:   {updates:,}")
    print(f"  No claim found (skipped):{no_claim:,}")

    if sample:
        print("\nSample recovery rates (up to 15):")
        print(f"  {'Case':50}  {'Claimed':>12}  {'Awarded':>12}  {'Rate':>7}  Class")
        for cid, cl, aw, rt, rc in sample:
            print(f"  {cid[:50]:<50}  ${cl:>11,.0f}  ${aw:>11,.0f}  {rt:>6.3f}  {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate civil recovery rates across case chunks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_aggregation(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
