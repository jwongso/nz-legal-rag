"""One-off script to fix url payloads in the nztt_moj collection.

Updates every point from the generic search URL to the NZLII decision URL.
"""

import re
import config
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

_WRONG_URL = "https://forms.justice.govt.nz/search/TT/"

client = QdrantClient(url=config.QDRANT_URL)
collection = config.QDRANT_TENANCY_COLLECTION

total = client.get_collection(collection).points_count
print(f"Total points: {total}")

fixed = 0
offset = None

while True:
    batch, next_offset = client.scroll(
        collection,
        limit=500,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    if not batch:
        break

    updates = {}
    for p in batch:
        if p.payload.get("url") != _WRONG_URL:
            continue
        case_id = p.payload.get("case_id", "")
        year = p.payload.get("year", 0)
        app = case_id.replace("NZTT-MOJ-", "")
        if year and app.isdigit():
            updates[p.id] = f"https://www.nzlii.org/nz/cases/NZTT/{year}/{app}.html"

    if updates:
        for point_id, url in updates.items():
            client.set_payload(
                collection,
                payload={"url": url},
                points=[point_id],
            )
        fixed += len(updates)

    if fixed % 10000 == 0 and fixed:
        print(f"  Fixed {fixed} points...")

    if next_offset is None:
        break
    offset = next_offset

print(f"Done: fixed {fixed} points")
