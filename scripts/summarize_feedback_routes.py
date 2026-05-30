#!/usr/bin/env python3
"""Summarize thumbs-down feedback by statute routing match status.

Reads feedback_full.jsonl and groups thumbs-down entries by matched routes.
Unmatched queries (no route triggered) are the long-tail routing candidates.

Usage:
  python scripts/summarize_feedback_routes.py
  python scripts/summarize_feedback_routes.py data/feedback_full.jsonl
  python scripts/summarize_feedback_routes.py --top 20
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_entries(path: Path) -> list[dict]:
    entries = []
    if not path.exists():
        print(f"[WARN] {path} not found.")
        return entries
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def summarize(path: Path, top_n: int) -> None:
    entries = load_entries(path)
    thumbs_down = [e for e in entries if e.get("rating") == -1]

    if not thumbs_down:
        print("No thumbs-down entries found.")
        return

    print(f"Total thumbs-down entries: {len(thumbs_down)}\n")

    unrouted_questions: list[str] = []
    route_counter: Counter = Counter()
    section_counter: Counter = Counter()

    for entry in thumbs_down:
        ctx = entry.get("context_debug") or {}
        sr = ctx.get("statute_routing") or {}
        matched = sr.get("matched_routes") or []
        forced = sr.get("forced_sections") or []
        question = entry.get("question", "").strip()

        if not matched:
            unrouted_questions.append(question)
        else:
            for r in matched:
                route_counter[r] += 1
            for s in forced:
                section_counter[s.replace("NZLEG/RTA/", "")] += 1

    print("=== Thumbs-down with NO route matched (long-tail candidates) ===")
    if not unrouted_questions:
        print("  None - all thumbs-down queries matched at least one route.")
    else:
        word_groups: Counter = Counter()
        for q in unrouted_questions:
            tokens = q.lower().split()
            for token in tokens:
                if len(token) > 4:
                    word_groups[token] += 1

        print(f"  {len(unrouted_questions)} unrouted thumbs-down queries:")
        for q in unrouted_questions[:top_n]:
            snippet = q[:120].replace("\n", " ")
            print(f"  - {snippet}")
        if len(unrouted_questions) > top_n:
            print(f"  ... and {len(unrouted_questions) - top_n} more")

        print("\n  Top recurring terms in unrouted queries (route candidates):")
        for term, count in word_groups.most_common(15):
            print(f"    {count:3d}x  {term}")

    print("\n=== Thumbs-down by route (where routing fired) ===")
    if not route_counter:
        print("  None.")
    else:
        for route, count in route_counter.most_common():
            print(f"  {count:3d}x  {route}")

    print("\n=== Thumbs-down by forced section injected ===")
    if not section_counter:
        print("  None.")
    else:
        for section, count in section_counter.most_common():
            print(f"  {count:3d}x  {section}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="data/feedback_full.jsonl",
        help="Path to feedback_full.jsonl (default: data/feedback_full.jsonl)",
    )
    parser.add_argument("--top", type=int, default=10, help="Max unrouted queries to show")
    args = parser.parse_args()
    summarize(Path(args.path), args.top)


if __name__ == "__main__":
    main()
