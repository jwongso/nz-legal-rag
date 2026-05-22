"""Context packing benchmark: measures how prompt format affects answer quality.

Retrieval is fixed (planner_filter_vector_legal, same hits for every format).
Only the context assembly changes between runs. This isolates the packing
variable from the retrieval variable.

Formats compared (see rag/context_packer.py for implementations):
  baseline      Current production: plain [N], 600-char truncation.
  metadata_rich Structured header (Title | Court | Date | Section), 600-char.
  full_chunk    Full chunk text (up to 1200 chars), plain [N].
  statute_first metadata_rich, NZLEG chunks first.
  top3_only     Top 3 docs, full text, metadata_rich.
  max2_per_doc  Up to 2 chunks per doc (10 chunks max), metadata_rich, 600-char.

Queries: general + statute task types from retrieval gold dataset (14 queries).
Sentencing and employment find-examples queries are excluded - they ask for a
list of cases rather than a substantive legal answer, making citation support
and answer completeness metrics unreliable.

Metrics per format:
  context_tokens    Approximate prompt token count (chars / 4)
  answer_tokens     Approximate answer token count
  has_citations     1 if answer contains at least one [N] reference
  all_in_context    1 if every [N] citation matches a source in the context
  no_context_flag   1 if answer contains "not enough" / "do not have" / similar
  source_diversity  Number of distinct [N] references in the answer
  latency_ms        Time from prompt send to answer receive

Run:
    python -m benchmarks.runners.run_context_packing
"""

import asyncio
import json
import re
import time
from pathlib import Path

import psycopg2

import config
from rag.context_packer import PackedContext, pack
from rag.court_planner import plan_courts
from rag.embedder import Embedder
from rag.legal_ranker import QueryContext, rerank as legal_rerank
from rag.retriever import VectorStore

_GOLD_PATH = Path("benchmarks/datasets/retrieval_gold.jsonl")
_REPORTS_DIR = Path("benchmarks/reports")
_FETCH_K = 50

_FORMATS = [
    "baseline",
    "metadata_rich",
    "full_chunk",
    "statute_first",
    "top3_only",
    "max2_per_doc",
]

_TASK_TYPES = {"general", "statute"}

_NO_CONTEXT_PATTERNS = re.compile(
    r"not enough (information|context|detail)|"
    r"(do|does|did) not (have|contain|include|provide)|"
    r"cannot (answer|determine|find)|"
    r"no (relevant|specific|sufficient)|"
    r"unable to (answer|find|determine)",
    re.IGNORECASE,
)

# Matches [N] where N is 1-2 digits only.
# NZ legal citation format uses 4-digit years: "[2023] NZCA 50" - these must NOT
# be counted as source citations. Paragraph numbers in judgments (e.g. [15]) CAN
# appear in answers but are rare and handled by the valid-range check below.
_CITATION_RE = re.compile(r"\[([1-9]\d?)\]")


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_citations(answer: str) -> set[int]:
    """Extract [N] source citation numbers, ignoring 4-digit years and zero."""
    return {int(m) for m in _CITATION_RE.findall(answer)}


def _all_in_context(cited: set[int], n_sources: int) -> bool:
    return bool(cited) and all(1 <= c <= n_sources for c in cited)


def _has_no_context(answer: str) -> bool:
    return bool(_NO_CONTEXT_PATTERNS.search(answer))


def _get_point_ids_for_courts(
    conn, courts: list[str], years: list[int] | None
) -> list[str]:
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


async def _retrieve_hits(
    query: str,
    query_vec: list[float],
    store: VectorStore,
    conn,
):
    """Planner-filtered retrieval. Returns raw (pre-dedup) hits for context packer."""
    plan = plan_courts(query)
    if plan.courts:
        point_ids = _get_point_ids_for_courts(conn, plan.courts, plan.years)
        if point_ids:
            raw = store.search_within(query_vec, point_ids, top_k=_FETCH_K)
        else:
            raw = store.search(query_vec, top_k=_FETCH_K)
    else:
        raw = store.search(query_vec, top_k=_FETCH_K)

    ctx = QueryContext.from_query(query)
    # Apply legal ranker on per-chunk basis before handing to context packer
    # (packer's dedup functions need score order to be correct)
    ordered = legal_rerank(raw, ctx)
    return ordered


async def _call_llm(context_block: str, sources: list[dict], question: str) -> tuple[str, float]:
    """Call the LLM with the packed context. Returns (answer, latency_ms)."""
    import httpx

    source_header = "\n".join(
        f"  [{i+1}] {s.get('title','Unknown')} | {s.get('court_name','')} | "
        f"{s.get('date','')} | {s.get('url','')}"
        for i, s in enumerate(sources)
    )
    user_message = (
        f"Source index:\n{source_header}\n\n"
        f"Context documents (numbered to match source index):\n\n{context_block}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        f"Answer using only the context above. Cite sources with [N] notation "
        f"matching the source index. After your answer, list every source you cited."
    )
    system_prompt = (
        "You are a legal research assistant specialising in New Zealand law.\n\n"
        "Rules:\n"
        "- Answer only from the provided context. Do not invent cases, statutes, or dates.\n"
        "- Always cite the source document for each claim (case name and year, or Act and section).\n"
        "- If the context does not contain enough information to answer, say so clearly.\n"
        "- Use plain English. Avoid legal jargon unless quoting directly from a source.\n"
        "- Do not give legal advice. Remind the user to consult a qualified NZ lawyer for their situation.\n"
    )
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": config.LLM_MAX_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient(base_url=config.LLM_BASE_URL, timeout=180) as client:
        resp = await client.post("/chat/completions", json=payload)
        if resp.status_code == 500:
            await asyncio.sleep(2)
            resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
    latency_ms = (time.monotonic() - t0) * 1000
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    return answer, latency_ms


def _score_answer(answer: str, packed: PackedContext) -> dict:
    cited = _extract_citations(answer)
    n_sources = len(packed.sources)
    valid = {c for c in cited if 1 <= c <= n_sources}
    spurious = cited - valid  # paragraph numbers not in source list
    return {
        "context_tokens": packed.token_estimate,
        "chunk_count": packed.chunk_count,
        "doc_count": packed.doc_count,
        "answer_tokens": _count_tokens(answer),
        "has_citations": 1 if valid else 0,
        "all_in_context": 1 if _all_in_context(cited, n_sources) else 0,
        "no_context_flag": 1 if _has_no_context(answer) else 0,
        "source_diversity": len(valid),
        "spurious_citations": len(spurious),
    }


async def run():
    gold = [
        json.loads(line)
        for line in _GOLD_PATH.read_text().splitlines()
        if json.loads(line).get("task_type") in _TASK_TYPES
    ]
    print(f"Context packing benchmark: {len(gold)} queries, {len(_FORMATS)} formats")
    print(f"Formats: {_FORMATS}\n")

    embedder = Embedder()
    store = VectorStore()
    conn = psycopg2.connect(dbname="nz_legal")

    # Results: list of {id, query, format -> {metrics + latency_ms}}
    results = []

    for qi, gold_rec in enumerate(gold, 1):
        qid = gold_rec["id"]
        query = gold_rec["query"]
        print(f"[{qi}/{len(gold)}] {qid}")
        print(f"  Q: {query[:80]}")

        query_vec = await embedder.embed(query)
        hits = await _retrieve_hits(query, query_vec, store, conn)

        qresult = {"id": qid, "query": query, "task_type": gold_rec["task_type"], "formats": {}}

        for fmt in _FORMATS:
            packed = pack(hits, fmt)
            answer, latency_ms = await _call_llm(packed.context_block, packed.sources, query)
            metrics = _score_answer(answer, packed)
            metrics["latency_ms"] = round(latency_ms)
            # Save raw answer for metric recomputation without re-running LLM
            metrics["answer_text"] = answer
            qresult["formats"][fmt] = metrics

            flag = "!" if metrics["no_context_flag"] else ""
            spur = f" spur={metrics['spurious_citations']}" if metrics["spurious_citations"] else ""
            print(
                f"  {fmt:14s}  ctx={metrics['context_tokens']:4d}tok  "
                f"ans={metrics['answer_tokens']:3d}tok  "
                f"cit={metrics['source_diversity']}  "
                f"in_ctx={metrics['all_in_context']}{spur}  "
                f"lat={latency_ms/1000:.1f}s {flag}"
            )
        results.append(qresult)

    _REPORTS_DIR.mkdir(exist_ok=True)
    report_path = _REPORTS_DIR / "context_packing.md"
    _write_report(results, report_path)
    json_path = _REPORTS_DIR / "context_packing.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nReports written to {_REPORTS_DIR}/")


def _write_report(results: list[dict], path: Path) -> None:
    lines = [
        "# Context Packing Benchmark",
        "",
        "Fixed retrieval: `planner_filter_vector_legal`. Only context assembly varies.",
        "Queries: general + statute task types (14 queries).",
        "Citation regex: `[1-9][0-9]?` only - 4-digit NZ case years ([2023] etc.) excluded.",
        "",
        "## Aggregate Metrics (mean across queries)",
        "",
        "| Format | ctx_tok | ans_tok | has_cit | all_in_ctx | no_ctx | diversity | spurious | lat_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    fmt_agg: dict[str, dict[str, list]] = {f: {} for f in _FORMATS}
    metric_keys = [
        "context_tokens", "answer_tokens", "has_citations", "all_in_context",
        "no_context_flag", "source_diversity", "spurious_citations", "latency_ms",
    ]
    for r in results:
        for fmt in _FORMATS:
            m = r["formats"].get(fmt, {})
            for k in metric_keys:
                fmt_agg[fmt].setdefault(k, []).append(m.get(k, 0))

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0

    for fmt in _FORMATS:
        agg = fmt_agg[fmt]
        ctx = mean(agg["context_tokens"])
        ans = mean(agg["answer_tokens"])
        cit = mean(agg["has_citations"])
        inc = mean(agg["all_in_context"])
        noc = mean(agg["no_context_flag"])
        div = mean(agg["source_diversity"])
        spu = mean(agg["spurious_citations"])
        lat = mean(agg["latency_ms"]) / 1000
        lines.append(
            f"| {fmt} | {ctx:.0f} | {ans:.0f} | {cit:.2f} | {inc:.2f} "
            f"| {noc:.2f} | {div:.1f} | {spu:.1f} | {lat:.1f} |"
        )

    lines += [
        "",
        "## Per-Query Detail",
        "",
        "| Query | Format | ctx_tok | ans_tok | has_cit | in_ctx | no_ctx | div | spur | lat_s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        for fmt in _FORMATS:
            m = r["formats"].get(fmt, {})
            lines.append(
                f"| {r['id']} | {fmt} "
                f"| {m.get('context_tokens',0)} "
                f"| {m.get('answer_tokens',0)} "
                f"| {m.get('has_citations',0)} "
                f"| {m.get('all_in_context',0)} "
                f"| {m.get('no_context_flag',0)} "
                f"| {m.get('source_diversity',0)} "
                f"| {m.get('spurious_citations',0)} "
                f"| {m.get('latency_ms',0)/1000:.1f} |"
            )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "| Metric | Description |",
        "|---|---|",
        "| ctx_tok | Approx prompt tokens for context block (chars / 4) |",
        "| ans_tok | Approx answer tokens |",
        "| has_cit | 1 if answer contains at least one valid source [N] citation |",
        "| all_in_ctx | 1 if every [N] citation (1-2 digits) is a valid source number |",
        "| no_ctx | 1 if answer says it lacks enough information |",
        "| diversity | Distinct valid [N] source citation indices in answer |",
        "| spurious | [N] patterns that are 1-2 digits but outside source range (para refs) |",
        "| lat_s | Total LLM call latency (seconds) |",
    ]

    path.write_text("\n".join(lines) + "\n")
    print(f"  -> {path}")


if __name__ == "__main__":
    asyncio.run(run())
