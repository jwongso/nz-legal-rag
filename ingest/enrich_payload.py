"""
Enrich existing Qdrant payloads with fields derived from already-stored data.

Adds without re-embedding or re-ingesting:
  - document_type   : 'decision' (all current content)
  - jurisdiction    : 'NZ' (all NZLII content is New Zealand jurisdiction)
  - legal_area      : derived from court code
                      criminal courts + sentencing payload -> 'criminal'
                      employment courts                   -> 'employment'
                      family court                        -> 'family'
                      environment court                   -> 'environment'
                      everything else                     -> 'civil'

Run once after ingestion. Safe to re-run - set_payload is idempotent.

Usage:
    python -m ingest.enrich_payload
    python -m ingest.enrich_payload --dry-run
    python -m ingest.enrich_payload --court NZCA   # single court only
"""

import argparse
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
import config

# Courts where legal_area depends on whether it's a criminal or civil matter.
# Use presence of sentencing payload as the signal.
_MIXED_COURTS = {"NZHC", "NZCA", "NZSC", "NZDC"}

_COURT_AREA: dict[str, str] = {
    "NZSC":    "civil",        # overridden to 'criminal' if sentencing present
    "NZCA":    "civil",
    "NZHC":    "civil",
    "NZDC":    "civil",
    "NZEmpC":  "employment",
    "NZERA":   "employment",
    "NZFC":    "family",
    "NZEnvC":  "environment",
    "NZACC":   "civil",
    "NZCorC":  "civil",
    "NZLCDT":  "civil",
    "NZHRRT":  "civil",
    "NZREADT": "civil",
    "NZTT":    "civil",
}


def _legal_area(payload: dict) -> str:
    court = payload.get("court", "")
    base = _COURT_AREA.get(court, "civil")
    if base == "civil" and court in _MIXED_COURTS:
        # Criminal matter: sentencing sub-dict present and has data
        sent = payload.get("sentencing", {})
        if isinstance(sent, dict) and sent.get("has_data"):
            return "criminal"
    return base


def run(dry_run: bool = False, court_filter: str | None = None) -> None:
    client = QdrantClient(url=config.QDRANT_URL)

    scroll_filter = None
    if court_filter:
        scroll_filter = Filter(must=[
            FieldCondition(key="court", match=MatchValue(value=court_filter))
        ])

    batch_size = 200
    offset = None
    total = 0
    updated = 0
    t0 = time.time()

    while True:
        results, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            with_payload=True,
            limit=batch_size,
            offset=offset,
        )
        if not results:
            break

        # Group by the derived values to batch set_payload calls
        # Each (document_type, jurisdiction, legal_area) combo -> list of point IDs
        groups: dict[tuple, list] = {}
        for r in results:
            p = r.payload
            # Skip if already enriched
            if all(k in p for k in ("document_type", "jurisdiction", "legal_area")):
                total += 1
                continue
            key = (
                p.get("document_type", "decision"),
                p.get("jurisdiction", "NZ"),
                _legal_area(p),
            )
            groups.setdefault(key, []).append(r.id)

        for (doc_type, jurisdiction, legal_area), ids in groups.items():
            if not dry_run:
                client.set_payload(
                    collection_name=config.QDRANT_COLLECTION,
                    payload={
                        "document_type": doc_type,
                        "jurisdiction":  jurisdiction,
                        "legal_area":    legal_area,
                    },
                    points=ids,
                )
            updated += len(ids)

        total += len(results)

        if total % 10000 < batch_size:
            elapsed = time.time() - t0
            print(f"  {total:,} scanned | {updated:,} updated | {elapsed:.0f}s", flush=True)

        if next_offset is None:
            break
        offset = next_offset

    elapsed = time.time() - t0
    print(f"\nDone {'(dry run) ' if dry_run else ''}")
    print(f"  Total scanned: {total:,}")
    print(f"  Points updated: {updated:,}")
    print(f"  Elapsed: {elapsed:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Qdrant payloads with derived fields.")
    parser.add_argument("--dry-run", action="store_true", help="Scan without writing")
    parser.add_argument("--court", help="Limit to a single court code (e.g. NZCA)")
    args = parser.parse_args()
    run(dry_run=args.dry_run, court_filter=args.court)


if __name__ == "__main__":
    main()
