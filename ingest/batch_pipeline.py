"""
Batch ingest pipeline with crash-resume for multi-court, multi-year coverage.

Tracks completed (court, year) pairs in data/ingest_progress.json.
Because Qdrant point IDs are deterministic UUID5 hashes of (case_id, chunk_index),
re-running a completed year is fully idempotent - no duplicate data.

Usage:
    # First batch: NZCA 1985-1989
    python -m ingest.batch_pipeline --courts NZCA --year-from 1985 --year-to 1989

    # Continue next batch without re-ingesting completed years
    python -m ingest.batch_pipeline --courts NZCA --year-from 1990 --year-to 1994

    # Multiple courts at once
    python -m ingest.batch_pipeline --courts NZCA NZHC --year-from 2000 --year-to 2005

    # Show what has been completed and what remains in a planned range
    python -m ingest.batch_pipeline --status --courts NZCA --year-from 1985 --year-to 2024

    # Dry run: show plan without ingesting
    python -m ingest.batch_pipeline --courts NZCA --year-from 1985 --year-to 1989 --dry-run

    # Remove a single (court, year) from the completed set so it re-runs
    python -m ingest.batch_pipeline --reset NZCA:1986

    # Wipe all progress (use with caution)
    python -m ingest.batch_pipeline --reset-all
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from ingest.pipeline import run as _ingest_run
from rag.embedder import Embedder
from rag.retriever import VectorStore

_PROGRESS_FILE = Path("data/ingest_progress.json")

_ALL_COURTS = [
    "NZSC", "NZCA", "NZHC", "NZDC",
    "NZEmpC", "NZERA", "NZFC", "NZEnvC",
    "NZACC", "NZCorC", "NZLCDT", "NZHRRT", "NZREADT", "NZTT",
]

# Recommended start years per court (earlier years have sparser NZLII coverage)
RECOMMENDED_START: dict[str, int] = {
    "NZSC":    2004,  # court only exists from 2004
    "NZCA":    1985,
    "NZHC":    2000,
    "NZDC":    2000,
    "NZEmpC":  2000,
    "NZERA":   2000,
    "NZFC":    2000,
    "NZEnvC":  1996,
    "NZACC":   2000,
    "NZCorC":  2000,
    "NZLCDT":  2000,
    "NZHRRT":  2000,
    "NZREADT": 2000,
    "NZTT":    2021,
}


# ---------------------------------------------------------------------------
# Progress file helpers
# ---------------------------------------------------------------------------

def _load_progress() -> dict:
    if _PROGRESS_FILE.exists():
        raw = json.loads(_PROGRESS_FILE.read_text())
        raw["completed"] = set(raw.get("completed", []))
        return raw
    return {"completed": set(), "log": []}


def _save_progress(progress: dict) -> None:
    _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = {**progress, "completed": sorted(progress["completed"])}
    _PROGRESS_FILE.write_text(json.dumps(out, indent=2))


def _mark_done(progress: dict, court: str, year: int, chunks: int) -> None:
    key = f"{court}:{year}"
    progress["completed"].add(key)
    progress.setdefault("log", []).append({
        "key": key, "chunks": chunks,
        "at": datetime.now().isoformat(timespec="seconds"),
    })
    _save_progress(progress)


# ---------------------------------------------------------------------------
# Core batch runner
# ---------------------------------------------------------------------------

async def _run_batch(
    courts: list[str],
    year_from: int,
    year_to: int,
    max_per_year: int,
    dry_run: bool,
) -> None:
    progress = _load_progress()
    plan = [
        (court, year)
        for court in courts
        for year in range(year_from, year_to + 1)
    ]

    pending = [(c, y) for c, y in plan if f"{c}:{y}" not in progress["completed"]]
    skipped = len(plan) - len(pending)

    print(f"Plan: {len(plan)} (court, year) pairs")
    print(f"  Already done: {skipped}")
    print(f"  To ingest:    {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    if dry_run:
        print("\nDry run - pairs that WOULD be ingested:")
        for court, year in pending:
            print(f"  {court} {year}")
        return

    # Load model and store once, reuse across all (court, year) pairs
    print("\nLoading embedding model...", flush=True)
    embedder = Embedder()
    store = VectorStore()
    store.ensure_collection()
    print("Model ready.\n", flush=True)

    total_chunks = 0
    errors: list[str] = []

    for i, (court, year) in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] === {court} {year} ===", flush=True)
        t0 = time.time()
        try:
            chunks = await _ingest_run(
                court, [year], max_per_year,
                _embedder=embedder, _store=store,
            )
            elapsed = time.time() - t0
            total_chunks += chunks
            _mark_done(progress, court, year, chunks)
            print(
                f"  -> {chunks} chunks in {elapsed:.0f}s "
                f"(total so far: {total_chunks:,})",
                flush=True,
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved - re-run to continue.", flush=True)
            sys.exit(0)
        except Exception as e:
            elapsed = time.time() - t0
            msg = f"{court}:{year} - {e}"
            errors.append(msg)
            print(f"  ERROR after {elapsed:.0f}s: {e}", flush=True)
            print("  Continuing with next pair...", flush=True)

    print(f"\n{'=' * 50}")
    print(f"Batch complete.")
    print(f"  New chunks indexed: {total_chunks:,}")
    print(f"  Errors:             {len(errors)}")
    for err in errors:
        print(f"    {err}")

    total_done = len(progress["completed"])
    print(f"  Total (court, year) pairs ever completed: {total_done}")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def _show_status(courts: list[str], year_from: int, year_to: int) -> None:
    progress = _load_progress()
    completed = progress["completed"]

    print(f"Progress file: {_PROGRESS_FILE}")
    print(f"Total completed pairs: {len(completed)}\n")

    for court in courts:
        done = [y for y in range(year_from, year_to + 1) if f"{court}:{y}" in completed]
        todo = [y for y in range(year_from, year_to + 1) if f"{court}:{y}" not in completed]
        print(f"{court}:")
        if done:
            runs = []
            start = done[0]
            for j in range(1, len(done)):
                if done[j] != done[j-1] + 1:
                    runs.append(f"{start}-{done[j-1]}" if start != done[j-1] else str(start))
                    start = done[j]
            runs.append(f"{start}-{done[-1]}" if start != done[-1] else str(start))
            print(f"  Done: {', '.join(runs)}")
        if todo:
            print(f"  Pending: {todo[0]}-{todo[-1]} ({len(todo)} years)")
        if not done and not todo:
            print("  (no years in range)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ingest with crash-resume. Progress saved to data/ingest_progress.json."
    )
    parser.add_argument(
        "--courts", nargs="+", default=["NZCA"],
        choices=_ALL_COURTS,
        help="Courts to ingest (default: NZCA)",
    )
    parser.add_argument("--year-from", type=int, default=1985, help="Start year (inclusive)")
    parser.add_argument("--year-to",   type=int, default=1989, help="End year (inclusive)")
    parser.add_argument("--max-per-year", type=int, default=500,
                        help="Max decisions per (court, year) (default: 500)")
    parser.add_argument("--threads", type=int, default=16,
                        help="CPU threads for embedding")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without ingesting")
    parser.add_argument("--status", action="store_true",
                        help="Show completion status for the given range and exit")
    parser.add_argument("--reset", metavar="COURT:YEAR",
                        help="Remove a single entry from the completed set (e.g. NZCA:1986)")
    parser.add_argument("--reset-all", action="store_true",
                        help="Wipe all progress (use with caution)")
    args = parser.parse_args()

    # Handle reset commands
    if args.reset_all:
        if _PROGRESS_FILE.exists():
            _PROGRESS_FILE.unlink()
        print("Progress wiped.")
        return

    if args.reset:
        progress = _load_progress()
        key = args.reset
        if key in progress["completed"]:
            progress["completed"].discard(key)
            _save_progress(progress)
            print(f"Removed {key} from completed set.")
        else:
            print(f"{key} was not in completed set.")
        return

    # Apply thread limit
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass

    if args.status:
        _show_status(args.courts, args.year_from, args.year_to)
        return

    asyncio.run(_run_batch(
        courts=args.courts,
        year_from=args.year_from,
        year_to=args.year_to,
        max_per_year=args.max_per_year,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
