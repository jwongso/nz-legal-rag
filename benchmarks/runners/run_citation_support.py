"""Citation support benchmark: measures whether cited sources back up the claims.

Extends the context packing benchmark by adding the 'supports_claim' dimension.
context_exists and in_context were already confirmed 1.00 in run_context_packing.py.
This benchmark asks the remaining question: does [N] actually support the sentence?

Approach:
  1. Load saved answers (baseline format) from benchmarks/reports/context_packing.json.
  2. Re-retrieve chunk texts for each query (planner pipeline, deterministic).
  3. Extract (claim_sentence, citation_idx) pairs from each answer.
  4. For each pair, ask the LLM to judge: YES / PARTIALLY / NO.
  5. Aggregate faithfulness rates per query and overall.

LLM judge prompt is tight (max_tokens=60) to keep latency low.
Estimated: ~14 queries x 4.2 avg citations = ~59 judge calls x 8s = ~8 minutes.

Run:
    python -m benchmarks.runners.run_citation_support
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psycopg2

import config
from rag.context_packer import pack
from rag.court_planner import plan_courts
from rag.embedder import Embedder
from rag.legal_ranker import QueryContext, rerank as legal_rerank
from rag.retriever import VectorStore

_GOLD_PATH = Path("benchmarks/datasets/retrieval_gold.jsonl")
_PACKING_JSON = Path("benchmarks/reports/context_packing.json")
_REPORTS_DIR = Path("benchmarks/reports")
_FETCH_K = 50
_TASK_TYPES = {"general", "statute"}

# Matches [1] through [99]; excludes 4-digit NZ case years
_CIT_RE = re.compile(r"\[([1-9]\d?)\]")

# Marker where the LLM starts listing sources - everything below is excluded
_SOURCES_SECTION_RE = re.compile(
    r"\n[*_-]*\s*\n?\s*(?:sources? cited|sources?|references?)[\s:*_-]*\n",
    re.IGNORECASE,
)

_VERDICT_RE = re.compile(r"\b(YES|PARTIALLY|NO)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Sentence / claim extraction
# ---------------------------------------------------------------------------

def _strip_sources_section(text: str) -> str:
    """Remove the trailing 'Sources Cited:' block from LLM answers."""
    m = _SOURCES_SECTION_RE.search(text)
    return text[: m.start()] if m else text


def _extract_claim_pairs(answer: str) -> list[tuple[str, int]]:
    """Return (claim_text, citation_idx) pairs from a LLM answer.

    Strategy: split the answer body into lines, find lines containing [N],
    capture the line as the claim, and associate each [N] on that line.
    One line may produce multiple pairs if it has multiple citations.
    De-duplicates by citation_idx, keeping the first (most specific) claim.
    """
    body = _strip_sources_section(answer)
    # Strip markdown emphasis markers for cleaner claim text
    body = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", body)

    seen_idx: set[int] = set()
    pairs: list[tuple[str, int]] = []

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        cits = [int(m) for m in _CIT_RE.findall(line)]
        if not cits:
            continue
        # Claim text: line with [N] markers stripped
        claim = _CIT_RE.sub("", line).strip(" .*-").strip()
        if len(claim) < 15:  # too short to be a meaningful claim
            continue
        for idx in cits:
            if idx not in seen_idx:
                seen_idx.add(idx)
                pairs.append((claim, idx))

    return pairs


# ---------------------------------------------------------------------------
# Retrieval helpers (mirrors run_context_packing._retrieve_hits)
# ---------------------------------------------------------------------------

def _get_point_ids_for_courts(conn, courts: list[str], years: list[int] | None) -> list[str]:
    cur = conn.cursor()
    if years:
        cur.execute(
            "SELECT c.qdrant_point_id FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE d.court = ANY(%s) AND d.decision_date IS NOT NULL "
            "AND EXTRACT(YEAR FROM d.decision_date) = ANY(%s)",
            (courts, years),
        )
    else:
        cur.execute(
            "SELECT c.qdrant_point_id FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE d.court = ANY(%s)",
            (courts,),
        )
    return [r[0] for r in cur.fetchall()]


async def _retrieve_hits(query: str, query_vec: list[float], store: VectorStore, conn):
    plan = plan_courts(query)
    if plan.courts:
        point_ids = _get_point_ids_for_courts(conn, plan.courts, plan.years)
        raw = store.search_within(query_vec, point_ids, top_k=_FETCH_K) if point_ids \
              else store.search(query_vec, top_k=_FETCH_K)
    else:
        raw = store.search(query_vec, top_k=_FETCH_K)
    ctx = QueryContext.from_query(query)
    return legal_rerank(raw, ctx)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a legal citation verifier. "
    "You will be given a claim from a legal research answer and the source passage it cites. "
    "Respond with exactly one word on the first line: YES, PARTIALLY, or NO. "
    "Then one sentence of explanation.\n\n"
    "YES = the passage directly supports the claim.\n"
    "PARTIALLY = the passage is related but only partially supports the claim.\n"
    "NO = the passage does not support the claim (or contradicts it)."
)


async def _judge(claim: str, source_text: str) -> tuple[str, str, float]:
    """Ask the LLM whether source_text supports claim. Returns (verdict, reason, latency_ms)."""
    user_msg = (
        f"Claim: {claim}\n\n"
        f"Source passage:\n{source_text[:800]}"
    )
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 80,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    # Fresh client per call: avoids keep-alive slot confusion when the server
    # drops a connection mid-request (observed with slow 35B inference runs).
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(base_url=config.LLM_BASE_URL, timeout=180) as client:
                resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            break
        except (httpx.RemoteProtocolError, httpx.TimeoutException, httpx.ConnectError,
                httpx.HTTPStatusError) as e:
            if attempt == 2:
                raise
            wait = 15 * (attempt + 1)
            print(f"  [retry {attempt + 1}/2 after {wait}s: {type(e).__name__}]")
            await asyncio.sleep(wait)
    latency_ms = (time.monotonic() - t0) * 1000
    text = resp.json()["choices"][0]["message"]["content"].strip()
    m = _VERDICT_RE.search(text)
    verdict = m.group(1).upper() if m else "UNKNOWN"
    reason = text.split("\n", 1)[1].strip() if "\n" in text else text
    return verdict, reason, latency_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class JudgementRecord:
    query_id: str
    claim: str
    citation_idx: int
    source_case_id: str
    source_text_snippet: str  # first 200 chars
    verdict: str
    reason: str
    latency_ms: float


async def run():
    # Load gold queries and packing answers
    gold = {
        r["id"]: r
        for line in _GOLD_PATH.read_text().splitlines()
        for r in [json.loads(line)]
        if r.get("task_type") in _TASK_TYPES
    }
    packing = {r["id"]: r for r in json.loads(_PACKING_JSON.read_text())}

    print(f"Citation support benchmark: {len(gold)} queries")
    print("Loading retrieval pipeline...")

    embedder = Embedder()
    store = VectorStore()
    conn = psycopg2.connect(dbname="nz_legal")

    all_records: list[JudgementRecord] = []

    for qi, (qid, gold_rec) in enumerate(gold.items(), 1):
        query = gold_rec["query"]
        answer = packing[qid]["formats"]["baseline"]["answer_text"]

        print(f"\n[{qi}/{len(gold)}] {qid}")
        print(f"  Q: {query[:80]}")

        # Re-retrieve chunks to get source texts (same deterministic pipeline)
        query_vec = await embedder.embed(query)
        hits = await _retrieve_hits(query, query_vec, store, conn)
        packed = pack(hits, "baseline")
        # Build correct source text lookup: best-scoring chunk per case_id.
        # Mirrors _dedup_one() inside the packer. Using hits[:N] is wrong
        # when a document has multiple chunks in the top hits, because a
        # lower-scoring chunk of an already-seen doc shifts all indices after it.
        best_by_case: dict[str, tuple[float, str]] = {}
        for h in hits:
            if h.case_id not in best_by_case or h.score > best_by_case[h.case_id][0]:
                best_by_case[h.case_id] = (h.score, h.text)
        source_chunks = [best_by_case[s["case_id"]][1] for s in packed.sources]

        pairs = _extract_claim_pairs(answer)
        print(f"  claims extracted: {len(pairs)}")

        for claim, idx in pairs:
            if idx < 1 or idx > len(source_chunks):
                # Out of range - should not happen after regex fix but guard anyway
                continue
            source_text = source_chunks[idx - 1]
            source_id = packed.sources[idx - 1]["case_id"]

            verdict, reason, lat = await _judge(claim, source_text)
            mark = {"YES": "+", "PARTIALLY": "~", "NO": "!"}.get(verdict, "?")
            print(f"  [{idx}] {mark} {verdict:9s}  {claim[:60]}")

            all_records.append(JudgementRecord(
                query_id=qid,
                claim=claim,
                citation_idx=idx,
                source_case_id=source_id,
                source_text_snippet=source_text[:200],
                verdict=verdict,
                reason=reason,
                latency_ms=round(lat),
            ))

    _write_report(all_records, gold)


def _write_report(records: list[JudgementRecord], gold: dict) -> None:
    _REPORTS_DIR.mkdir(exist_ok=True)

    # Save raw judgements as JSON
    raw_path = _REPORTS_DIR / "citation_support.json"
    raw_path.write_text(json.dumps(
        [r.__dict__ for r in records], indent=2
    ))

    # Compute aggregates
    total = len(records)
    yes = sum(1 for r in records if r.verdict == "YES")
    partial = sum(1 for r in records if r.verdict == "PARTIALLY")
    no = sum(1 for r in records if r.verdict == "NO")
    unknown = total - yes - partial - no

    faithful = yes / total if total else 0
    partial_r = partial / total if total else 0
    unsupported = no / total if total else 0

    lines = [
        "# Citation Support Benchmark",
        "",
        "Measures whether cited sources actually back up the claims in the generated answer.",
        "Format: `baseline` (current production). LLM judge: Qwen3 (temperature=0).",
        "Queries: general + statute task types (14 queries).",
        "",
        "## Overall Verdict",
        "",
        f"| Verdict | Count | Rate |",
        f"|---|---:|---:|",
        f"| YES (passage directly supports claim) | {yes} | {faithful:.2f} |",
        f"| PARTIALLY (related but incomplete support) | {partial} | {partial_r:.2f} |",
        f"| NO (passage does not support claim) | {no} | {unsupported:.2f} |",
        f"| UNKNOWN (judge parse failure) | {unknown} | {unknown/total if total else 0:.2f} |",
        f"| **Total claim-citation pairs** | **{total}** | |",
        "",
    ]

    # Per-query summary
    by_query: dict[str, list[JudgementRecord]] = {}
    for r in records:
        by_query.setdefault(r.query_id, []).append(r)

    lines += [
        "## Per-Query Summary",
        "",
        "| Query | pairs | YES | PARTIAL | NO | faithful |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for qid, recs in by_query.items():
        n = len(recs)
        y = sum(1 for r in recs if r.verdict == "YES")
        p = sum(1 for r in recs if r.verdict == "PARTIALLY")
        nn = sum(1 for r in recs if r.verdict == "NO")
        lines.append(f"| {qid} | {n} | {y} | {p} | {nn} | {y/n:.2f} |")

    # Per-record detail
    lines += [
        "",
        "## Judgement Detail",
        "",
        "| Query | [N] | Source | Verdict | Claim (truncated) | Reason |",
        "|---|---:|---|---|---|---|",
    ]
    for r in records:
        claim_t = r.claim[:60].replace("|", "/")
        reason_t = r.reason[:80].replace("|", "/")
        src_t = r.source_case_id
        lines.append(
            f"| {r.query_id} | [{r.citation_idx}] | {src_t} "
            f"| {r.verdict} | {claim_t} | {reason_t} |"
        )

    md_path = _REPORTS_DIR / "citation_support.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\n--- Citation Support Summary ---")
    print(f"  Total pairs judged: {total}")
    print(f"  YES (faithful):   {yes:3d} / {total} = {faithful:.2f}")
    print(f"  PARTIALLY:        {partial:3d} / {total} = {partial_r:.2f}")
    print(f"  NO (unsupported): {no:3d} / {total} = {unsupported:.2f}")
    print(f"\n  -> {md_path}")
    print(f"  -> {raw_path}")


if __name__ == "__main__":
    asyncio.run(run())
