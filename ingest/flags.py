"""
Keyword-based flagging for NZ legal cases.

Flags are stored as a list in the Qdrant payload field 'flags'.
This is a coarse first pass - high recall, moderate precision.
LLM-based refinement happens at query time in the /notable endpoint.

Flags are set at the chunk level so retrieval returns the specific
chunk that contains the flagged content, not just any chunk from the case.

Usage:
    python -m ingest.flags          # backfill all existing chunks
    python -m ingest.flags --dry-run  # count matches without writing
"""

import argparse
import re

from qdrant_client import QdrantClient

import config

# Each key is a flag name. Value is list of regex patterns (any match = flag set).
# Patterns are case-insensitive. Kept specific enough to avoid noise.
FLAG_PATTERNS: dict[str, list[str]] = {
    # --- Criminal defences ---
    "self_defence": [
        r"self.defenc",
        r"justified in using.{0,50}force",
        r"defenc\w+ of (?:him|her|them)self",
        r"\bs\.?\s*48.{0,20}Crimes Act",
    ],
    "provocation": [
        r"\bprovocation\b",
        r"loss of self.control",
    ],
    "diminished_responsibility": [
        r"diminished responsibility",
        r"mental impairment defence",
        r"not guilty by reason of insanity",
        r"unfit to (?:stand trial|plead)",
        r"fitness to stand trial",
    ],
    "necessity": [
        r"defence of necessity",
        r"necessity as a defence",
        r"act(?:ed|ing) out of necessity",
    ],
    "duress": [
        r"\bduress\b",
    ],

    # --- Mitigating / special factors ---
    "mental_health": [
        r"mental health (?:evidence|issue|assessment|report|condition|disorder|history)",
        r"psychiatric (?:report|assessment|evidence|disorder|condition|history)",
        r"\bPTSD\b",
        r"post.traumatic stress",
        r"mental disorder",
        r"intellectual disability",
        r"cognitive impairment",
    ],
    "intoxication": [
        r"\bintoxication\b",
        r"voluntar\w+ intoxicat",
        r"extreme intoxication",
    ],
    "youth": [
        r"\byoung (?:person|offender)\b",
        r"youth (?:court|justice|offend)",
        r"Oranga Tamariki",
        r"Children.Young Persons",
        r"\bjuvenile\b",
    ],
    "tikanga_maori": [
        r"\btikanga\b",
        r"kaupapa M.ori",
        r"Te Tiriti o Waitangi",
        r"Treaty of Waitangi",
        r"customary (?:law|rights|title|land)",
        r"\bwhanau\b",
        r"\bhapu\b",
        r"\biwi\b",
    ],
    "cultural_factors": [
        r"cultural (?:background|factors|consideration|evidence|context|practice|tradition)",
    ],

    # --- Legal anomalies ---
    "novel_argument": [
        r"novel (?:point|question|issue|argument|approach)",
        r"first (?:time|occasion) (?:this )?(?:court|question|issue)",
        r"not previously (?:been )?(?:considered|decided|determined|addressed)",
        r"question of first impression",
        r"no (?:New Zealand |NZ )?authority (?:on|for|addressing)",
        r"\bunprecedented\b",
        r"no (?:directly )?applicable (?:precedent|authority)",
    ],
    "jurisdictional_challenge": [
        r"jurisdictional (?:challenge|issue|error|question|basis)",
        r"outside (?:the )?jurisdiction",
        r"lacks? (?:the )?jurisdiction",
        r"excess of (?:its )?jurisdiction",
        r"no jurisdiction to",
    ],
    "procedural_irregularity": [
        r"breach of natural justice",
        r"procedural (?:fairness|failure|irregularity|error)",
        r"\bultra vires\b",
        r"denied (?:a|the) (?:right to be heard|opportunity to be heard)",
        r"apparent bias",
        r"reasonable apprehension of bias",
        r"procedurally unfair",
    ],

    # --- Outcome features ---
    "exemplary_damages": [
        r"exemplary damages",
        r"punitive damages",
        r"aggravated damages",
    ],
    "contempt": [
        r"contempt of court",
        r"found (?:to be )?in contempt",
        r"sentenced for contempt",
    ],
    "suppressed_identity": [
        r"\[Suppressed\]",
        r"name suppression",
        r"suppression order",
        r"interim suppression",
        r"permanent suppression",
    ],
    "whistleblower": [
        r"whistleblower",
        r"protected disclosure",
        r"Protected Disclosures Act",
    ],
    "lack_of_motive": [
        r"no (?:apparent |clear |obvious )?motive",
        r"absence of (?:any )?motive",
        r"lack of motive",
        r"motive (?:is|was|remains) (?:unclear|unknown|not established|unexplained)",
        r"could not establish (?:a )?motive",
    ],
    "self_represented": [
        r"self.represented",
        r"litigant in person",
        r"appearing (?:for himself|for herself|in person|without counsel)",
        r"unrepresented (?:party|appellant|defendant|plaintiff)",
    ],
}

# Human-readable labels for UI display
FLAG_LABELS: dict[str, str] = {
    "self_defence": "Self-defence",
    "provocation": "Provocation",
    "diminished_responsibility": "Diminished responsibility",
    "necessity": "Necessity defence",
    "duress": "Duress",
    "mental_health": "Mental health",
    "intoxication": "Intoxication",
    "youth": "Youth / young person",
    "tikanga_maori": "Tikanga Maori",
    "cultural_factors": "Cultural factors",
    "novel_argument": "Novel legal argument",
    "jurisdictional_challenge": "Jurisdictional challenge",
    "procedural_irregularity": "Procedural irregularity",
    "exemplary_damages": "Exemplary damages",
    "contempt": "Contempt of court",
    "suppressed_identity": "Suppressed identity",
    "whistleblower": "Whistleblower / protected disclosure",
    "lack_of_motive": "Lack of motive",
    "self_represented": "Self-represented party",
}

_COMPILED: dict[str, list[re.Pattern]] = {
    flag: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]
    for flag, patterns in FLAG_PATTERNS.items()
}


def detect_flags(text: str) -> list[str]:
    """Return sorted list of flag names matching the given text."""
    return sorted(
        flag for flag, patterns in _COMPILED.items()
        if any(p.search(text) for p in patterns)
    )


def backfill_flags(dry_run: bool = False, batch_size: int = 500) -> None:
    """Scan all Qdrant chunks and write flags to payload."""
    client = QdrantClient(url=config.QDRANT_URL)

    if not dry_run:
        try:
            client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION,
                field_name="flags",
                field_schema="keyword",
            )
            print("Created payload index on 'flags' field.")
        except Exception:
            print("Payload index on 'flags' already exists.")

    total = 0
    flagged = 0
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

        # Group points by flag combination for batched set_payload
        groups: dict[str, list] = {}
        for point in points:
            text = point.payload.get("text", "")
            title = point.payload.get("title", "")
            flags = detect_flags(f"{title} {text}")
            key = "|".join(flags)
            groups.setdefault(key, []).append(point.id)
            if flags:
                flagged += 1
                for f in flags:
                    flag_counts[f] = flag_counts.get(f, 0) + 1

        if not dry_run:
            for key, ids in groups.items():
                flags = [f for f in key.split("|") if f]
                client.set_payload(
                    collection_name=config.QDRANT_COLLECTION,
                    payload={"flags": flags},
                    points=ids,
                )

        total += len(points)
        if total % 5000 == 0 or next_offset is None:
            print(f"  {total:,} processed, {flagged:,} flagged...")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Done.")
    print(f"  Total chunks: {total:,}")
    print(f"  Flagged chunks: {flagged:,} ({100*flagged/total:.1f}%)")
    print("\nFlag breakdown:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        label = FLAG_LABELS.get(flag, flag)
        print(f"  {label:<40} {count:>6,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill flags on existing Qdrant chunks")
    parser.add_argument("--dry-run", action="store_true", help="Count matches without writing")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    backfill_flags(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
