"""Context packing: assembles retrieved hits into a prompt context block.

Six formats benchmarked in benchmarks/runners/run_context_packing.py:

  baseline      Current production: one chunk/doc, 600-char truncation, plain [N] prefix.
  metadata_rich One chunk/doc, 600-char truncation, structured header per chunk.
  full_chunk    One chunk/doc, full text (no truncation), no metadata header.
  statute_first metadata_rich ordering with NZLEG chunks sorted before case chunks.
  top3_only     Top 3 docs, full text, metadata_rich headers.
  max2_per_doc  Up to 2 chunks per doc (top 5 docs), metadata_rich, 600-char each.

Each format returns a (context_block, sources, token_estimate) tuple.
token_estimate is approximate (chars / 4) for latency budgeting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.retriever import SearchResult

_TRUNC = 600   # baseline/metadata_rich truncation limit (chars)
_TRUNC_WIDE = 1200  # full_chunk is the full text; this is a fallback safety cap


@dataclass
class PackedContext:
    context_block: str        # assembled text for the prompt
    sources: list[dict]       # [{case_id, title, court_name, date, url}]
    token_estimate: int       # approx prompt tokens (chars / 4)
    chunk_count: int          # number of chunks in context_block
    doc_count: int            # number of unique documents


def _source_dict(h: SearchResult) -> dict:
    return {
        "case_id": h.case_id,
        "title": h.title,
        "court_name": h.court_name,
        "date": h.date,
        "url": h.url,
    }


def _meta_header(h: SearchResult, idx: int) -> str:
    parts = [h.title or h.case_id]
    if h.court_name:
        parts.append(h.court_name)
    if h.date:
        parts.append(h.date)
    heading = h.payload.get("section_heading", "")
    if heading:
        parts.append(heading)
    return f"[{idx}] {' | '.join(parts)}"


def _dedup_one(hits: list[SearchResult], top_docs: int = 5) -> list[SearchResult]:
    """One best-scoring chunk per document, capped at top_docs."""
    seen: dict[str, SearchResult] = {}
    for h in hits:
        if h.case_id not in seen or h.score > seen[h.case_id].score:
            seen[h.case_id] = h
    by_score = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return by_score[:top_docs]


def _dedup_max2(hits: list[SearchResult], top_docs: int = 5) -> list[SearchResult]:
    """Up to 2 best chunks per document, capped at top_docs unique documents."""
    seen: dict[str, list[SearchResult]] = {}
    for h in hits:
        if h.case_id not in seen:
            seen[h.case_id] = [h]
        elif len(seen[h.case_id]) < 2 and h not in seen[h.case_id]:
            seen[h.case_id].append(h)
    # sort documents by best chunk score, then flatten
    doc_order = sorted(seen.keys(), key=lambda k: seen[k][0].score, reverse=True)
    result = []
    for cid in doc_order[:top_docs]:
        result.extend(seen[cid])
    return result


def _is_statute(h: SearchResult) -> bool:
    return h.payload.get("court", "") == "NZLEG"


def pack(hits: list[SearchResult], fmt: str) -> PackedContext:
    """Assemble context block for the given format name."""
    if fmt == "baseline":
        return _baseline(hits)
    if fmt == "metadata_rich":
        return _metadata_rich(hits)
    if fmt == "full_chunk":
        return _full_chunk(hits)
    if fmt == "statute_first":
        return _statute_first(hits)
    if fmt == "top3_only":
        return _top3_only(hits)
    if fmt == "max2_per_doc":
        return _max2_per_doc(hits)
    raise ValueError(f"Unknown context format: {fmt!r}")


# ---------------------------------------------------------------------------
# Format implementations
# ---------------------------------------------------------------------------

def _baseline(hits: list[SearchResult]) -> PackedContext:
    """Current production format: plain [N] prefix, 600-char truncation."""
    docs = _dedup_one(hits)
    parts = []
    sources = []
    for i, h in enumerate(docs, 1):
        text = h.text[:_TRUNC]
        parts.append(f"[{i}] {text}")
        sources.append(_source_dict(h))
    block = "\n\n---\n\n".join(parts)
    return PackedContext(block, sources, len(block) // 4, len(parts), len(docs))


def _metadata_rich(hits: list[SearchResult]) -> PackedContext:
    """Structured header per chunk: [N] Title | Court | Date | Section."""
    docs = _dedup_one(hits)
    parts = []
    sources = []
    for i, h in enumerate(docs, 1):
        header = _meta_header(h, i)
        text = h.text[:_TRUNC]
        parts.append(f"{header}\n\n{text}")
        sources.append(_source_dict(h))
    block = "\n\n---\n\n".join(parts)
    return PackedContext(block, sources, len(block) // 4, len(parts), len(docs))


def _full_chunk(hits: list[SearchResult]) -> PackedContext:
    """Full chunk text, no truncation, plain [N] prefix."""
    docs = _dedup_one(hits)
    parts = []
    sources = []
    for i, h in enumerate(docs, 1):
        text = h.text[:_TRUNC_WIDE]
        parts.append(f"[{i}] {text}")
        sources.append(_source_dict(h))
    block = "\n\n---\n\n".join(parts)
    return PackedContext(block, sources, len(block) // 4, len(parts), len(docs))


def _statute_first(hits: list[SearchResult]) -> PackedContext:
    """metadata_rich ordering with NZLEG chunks sorted before case chunks."""
    docs = _dedup_one(hits)
    statute = [h for h in docs if _is_statute(h)]
    cases = [h for h in docs if not _is_statute(h)]
    ordered = statute + cases
    parts = []
    sources = []
    for i, h in enumerate(ordered, 1):
        header = _meta_header(h, i)
        text = h.text[:_TRUNC]
        parts.append(f"{header}\n\n{text}")
        sources.append(_source_dict(h))
    block = "\n\n---\n\n".join(parts)
    return PackedContext(block, sources, len(block) // 4, len(parts), len(ordered))


def _top3_only(hits: list[SearchResult]) -> PackedContext:
    """Top 3 docs only, full text, metadata_rich headers."""
    docs = _dedup_one(hits, top_docs=3)
    parts = []
    sources = []
    for i, h in enumerate(docs, 1):
        header = _meta_header(h, i)
        text = h.text[:_TRUNC_WIDE]
        parts.append(f"{header}\n\n{text}")
        sources.append(_source_dict(h))
    block = "\n\n---\n\n".join(parts)
    return PackedContext(block, sources, len(block) // 4, len(parts), len(docs))


def _max2_per_doc(hits: list[SearchResult]) -> PackedContext:
    """Up to 2 chunks per doc (top 5 docs), metadata_rich, 600-char each."""
    chunks = _dedup_max2(hits, top_docs=5)
    seen_docs: list[str] = []
    doc_idx: dict[str, int] = {}
    sources = []
    for h in chunks:
        if h.case_id not in doc_idx:
            seen_docs.append(h.case_id)
            doc_idx[h.case_id] = len(seen_docs)
            sources.append(_source_dict(h))

    parts = []
    chunk_counters: dict[str, int] = {}
    for h in chunks:
        n = doc_idx[h.case_id]
        sub = chunk_counters.get(h.case_id, 0) + 1
        chunk_counters[h.case_id] = sub
        label = f"[{n}{'b' if sub == 2 else ''}]"
        header = _meta_header(h, n)
        # re-label header with sub-chunk marker
        header = re.sub(r"^\[\d+\]", label, header)
        text = h.text[:_TRUNC]
        parts.append(f"{header}\n\n{text}")

    block = "\n\n---\n\n".join(parts)
    return PackedContext(block, sources, len(block) // 4, len(parts), len(seen_docs))
