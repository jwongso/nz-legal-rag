"""Answer quality benchmark: faithfulness, completeness, no-context accuracy.

Extends the citation support benchmark by evaluating the answer holistically
rather than one citation at a time.

Three dimensions per query:
  faithfulness   1-5  Does the answer only claim facts present in the sources?
  completeness   1-5  Does the answer cover all key legal points the question needs?
  gaps           list Free-text list of important points that are missing or wrong.

For queries where baseline answered "not enough context" (no_context_flag=1),
an additional judge call checks whether that claim was justified given the context.

Data source: baseline answers from benchmarks/reports/context_packing.json.
Context:     re-retrieved live (same deterministic pipeline, no new LLM generation).

Scoring guide passed to judge:
  5 = complete / fully faithful
  4 = mostly complete / mostly faithful, minor omission or unsupported aside
  3 = partially complete / partially faithful, a key point missing or one invented claim
  2 = incomplete / unfaithful, major legal point missing or clearly invented
  1 = very incomplete / hallucinating, does not substantively answer the question

Run:
    python -m benchmarks.runners.run_answer_quality
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

_SCORE_RE = re.compile(r"\b([1-5])\b")
_GAPS_RE = re.compile(
    r"(?:gaps?|missing|omissions?|incomplete)\s*[:\-]?\s*(.+?)(?=\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Retrieval helper (same pipeline as run_citation_support)
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
# Judge prompts
# ---------------------------------------------------------------------------

_QUALITY_SYSTEM = """You are an expert reviewer of AI-generated legal research answers for New Zealand law.

You will be given:
  - A legal question
  - The source context the AI had access to (numbered [1] to [N])
  - The AI's answer

Rate the answer on TWO dimensions using the scale below:

FAITHFULNESS (1-5): Does the answer only state facts that are actually present in the provided sources?
  5 = Every claim is directly traceable to a source passage.
  4 = Almost all claims are traceable; one minor unsupported aside.
  3 = Most claims are traceable; one claim goes noticeably beyond the sources.
  2 = Several claims are not supported or go beyond what the sources say.
  1 = The answer fabricates legal rules, case names, or dates not in the sources.

COMPLETENESS (1-5): Does the answer cover all key legal points the question requires?
  5 = All key points covered; answer is substantively complete.
  4 = Most key points covered; one minor point omitted.
  3 = Core points covered; one significant point missing.
  2 = Only partially addresses the question; a major aspect is missing.
  1 = Answer is superficial or almost entirely fails to address the question.

Your response MUST use this exact format:
FAITHFULNESS: <score>
COMPLETENESS: <score>
GAPS: <bullet list of missing or fabricated points, or "none" if score is 4-5 on both>"""


_NOCTX_SYSTEM = """You are a reviewer checking whether an AI legal assistant correctly identified a gap in its source material.

You will be given:
  - A legal question
  - The source context the AI had access to
  - The AI's answer, which claims it does not have enough information

Your task: read the source context and determine whether the context actually DOES contain enough information
to answer the question meaningfully.

Respond with exactly:
JUSTIFIED: YES or NO
REASON: <one sentence explaining your judgment>"""


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _source_header(sources: list[dict]) -> str:
    return "\n".join(
        f"  [{i+1}] {s.get('title', 'Unknown')} | {s.get('court_name', '')} "
        f"| {s.get('date', '')} | {s.get('url', '')}"
        for i, s in enumerate(sources)
    )


async def _judge_quality(
    question: str,
    context_block: str,
    sources: list[dict],
    answer: str,
) -> tuple[int, int, str, float]:
    """Rate faithfulness and completeness. Returns (faith, complete, gaps, latency_ms)."""
    # Reconstruct the full context the LLM originally saw: source index + context block.
    # Passing only context_block causes the judge to flag source titles (which come from
    # the source index) as "fabricated" because they don't appear in the raw chunk text.
    full_context = (
        f"Source index:\n{_source_header(sources)}\n\n"
        f"Context documents:\n\n{context_block}"
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"Source context:\n{full_context[:3500]}\n\n"
        f"Answer to evaluate:\n{answer}"
    )
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": _QUALITY_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(base_url=config.LLM_BASE_URL, timeout=300) as client:
                resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            break
        except (httpx.RemoteProtocolError, httpx.TimeoutException,
                httpx.ConnectError, httpx.HTTPStatusError) as e:
            if attempt == 2:
                raise
            wait = 20 * (attempt + 1)
            print(f"  [retry {attempt + 1}/2 after {wait}s: {type(e).__name__}]")
            await asyncio.sleep(wait)
    latency_ms = (time.monotonic() - t0) * 1000
    text = resp.json()["choices"][0]["message"]["content"].strip()

    faith = _parse_score(text, "FAITHFULNESS")
    complete = _parse_score(text, "COMPLETENESS")
    gaps = _parse_gaps(text)
    return faith, complete, gaps, latency_ms


async def _judge_noctx(
    question: str,
    context_block: str,
    sources: list[dict],
    answer: str,
) -> tuple[str, str, float]:
    """Check if 'not enough context' claim was justified. Returns (justified, reason, latency_ms)."""
    full_context = (
        f"Source index:\n{_source_header(sources)}\n\n"
        f"Context documents:\n\n{context_block}"
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"Source context:\n{full_context[:3500]}\n\n"
        f"AI answer:\n{answer}"
    )
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": _NOCTX_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 120,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(base_url=config.LLM_BASE_URL, timeout=300) as client:
                resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            break
        except (httpx.RemoteProtocolError, httpx.TimeoutException,
                httpx.ConnectError, httpx.HTTPStatusError) as e:
            if attempt == 2:
                raise
            wait = 20 * (attempt + 1)
            print(f"  [retry {attempt + 1}/2 after {wait}s: {type(e).__name__}]")
            await asyncio.sleep(wait)
    latency_ms = (time.monotonic() - t0) * 1000
    text = resp.json()["choices"][0]["message"]["content"].strip()
    justified = "YES" if re.search(r"JUSTIFIED\s*:\s*YES", text, re.IGNORECASE) else "NO"
    reason_m = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE)
    reason = reason_m.group(1).strip() if reason_m else text
    return justified, reason, latency_ms


def _parse_score(text: str, label: str) -> int:
    m = re.search(rf"{label}\s*:\s*([1-5])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0  # parse failure


def _parse_gaps(text: str) -> str:
    m = re.search(r"GAPS\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:400]
    return ""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QualityRecord:
    query_id: str
    query: str
    faithfulness: int
    completeness: int
    gaps: str
    no_context_flag: int
    noctx_justified: str  # "YES", "NO", or "N/A"
    noctx_reason: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run():
    gold = {
        r["id"]: r
        for line in _GOLD_PATH.read_text().splitlines()
        for r in [json.loads(line)]
        if r.get("task_type") in _TASK_TYPES
    }
    packing = {r["id"]: r for r in json.loads(_PACKING_JSON.read_text())}

    print(f"Answer quality benchmark: {len(gold)} queries")
    print("Loading retrieval pipeline...")

    embedder = Embedder()
    store = VectorStore()
    conn = psycopg2.connect(dbname="nz_legal")

    records: list[QualityRecord] = []

    for qi, (qid, gold_rec) in enumerate(gold.items(), 1):
        query = gold_rec["query"]
        baseline = packing[qid]["formats"]["baseline"]
        answer = baseline["answer_text"]
        no_ctx = baseline["no_context_flag"]

        print(f"\n[{qi}/{len(gold)}] {qid}")
        print(f"  Q: {query[:80]}")

        query_vec = await embedder.embed(query)
        hits = await _retrieve_hits(query, query_vec, store, conn)
        packed = pack(hits, "baseline")
        context_block = packed.context_block
        sources = packed.sources

        faith, complete, gaps, lat = await _judge_quality(query, context_block, sources, answer)
        faith_s = str(faith) if faith else "?"
        comp_s = str(complete) if complete else "?"
        print(f"  faithfulness={faith_s}/5  completeness={comp_s}/5  lat={lat/1000:.1f}s")
        if gaps and gaps.lower() != "none":
            print(f"  gaps: {gaps[:120]}")

        noctx_justified = "N/A"
        noctx_reason = ""
        if no_ctx:
            print(f"  [no_context_flag=1 - checking if justified]")
            noctx_justified, noctx_reason, noctx_lat = await _judge_noctx(
                query, context_block, sources, answer
            )
            print(f"  justified={noctx_justified}: {noctx_reason[:100]}")
            lat += noctx_lat

        records.append(QualityRecord(
            query_id=qid,
            query=query,
            faithfulness=faith,
            completeness=complete,
            gaps=gaps,
            no_context_flag=no_ctx,
            noctx_justified=noctx_justified,
            noctx_reason=noctx_reason,
            latency_ms=round(lat),
        ))

    _write_report(records)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(records: list[QualityRecord]) -> None:
    _REPORTS_DIR.mkdir(exist_ok=True)

    raw_path = _REPORTS_DIR / "answer_quality.json"
    raw_path.write_text(json.dumps([r.__dict__ for r in records], indent=2))

    total = len(records)
    valid = [r for r in records if r.faithfulness > 0 and r.completeness > 0]
    avg_faith = sum(r.faithfulness for r in valid) / len(valid) if valid else 0
    avg_comp = sum(r.completeness for r in valid) / len(valid) if valid else 0

    faith_dist = {i: sum(1 for r in valid if r.faithfulness == i) for i in range(1, 6)}
    comp_dist = {i: sum(1 for r in valid if r.completeness == i) for i in range(1, 6)}

    noctx_records = [r for r in records if r.no_context_flag]
    noctx_false = sum(1 for r in noctx_records if r.noctx_justified == "NO")

    lines = [
        "# Answer Quality Benchmark",
        "",
        "Evaluates AI-generated answers holistically on faithfulness and completeness.",
        "Format: `baseline` (current production). LLM judge: Qwen3.6-35B-A3B (temperature=0).",
        "Queries: general + statute task types (14 queries).",
        "",
        "## Overall Scores",
        "",
        f"| Dimension | Mean (1-5) | Dist: 5 | 4 | 3 | 2 | 1 |",
        f"|---|---:|---:|---:|---:|---:|---:|",
        f"| Faithfulness | {avg_faith:.2f} "
        f"| {faith_dist[5]} | {faith_dist[4]} | {faith_dist[3]} | {faith_dist[2]} | {faith_dist[1]} |",
        f"| Completeness | {avg_comp:.2f} "
        f"| {comp_dist[5]} | {comp_dist[4]} | {comp_dist[3]} | {comp_dist[2]} | {comp_dist[1]} |",
        "",
    ]

    if noctx_records:
        lines += [
            "## No-Context Flag Accuracy",
            "",
            f"Queries where baseline said 'not enough context': {len(noctx_records)}",
            f"Verified unjustified (context had the answer): {noctx_false}",
            "",
            "| Query | Justified | Reason |",
            "|---|---|---|",
        ]
        for r in noctx_records:
            lines.append(f"| {r.query_id} | {r.noctx_justified} | {r.noctx_reason[:100]} |")
        lines.append("")

    lines += [
        "## Per-Query Scores",
        "",
        "| Query | Faith | Complete | no_ctx | Gaps |",
        "|---|---:|---:|---:|---|",
    ]
    for r in records:
        faith_s = str(r.faithfulness) if r.faithfulness else "?"
        comp_s = str(r.completeness) if r.completeness else "?"
        gaps_t = r.gaps[:80].replace("|", "/").replace("\n", " ") if r.gaps else "none"
        lines.append(
            f"| {r.query_id} | {faith_s} | {comp_s} | {r.no_context_flag} | {gaps_t} |"
        )

    md_path = _REPORTS_DIR / "answer_quality.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\n--- Answer Quality Summary ---")
    print(f"  Faithfulness mean: {avg_faith:.2f} / 5")
    print(f"  Completeness mean: {avg_comp:.2f} / 5")
    if noctx_records:
        print(f"  No-ctx queries: {len(noctx_records)}, unjustified: {noctx_false}")
    print(f"\n  -> {md_path}")
    print(f"  -> {raw_path}")


if __name__ == "__main__":
    asyncio.run(run())
