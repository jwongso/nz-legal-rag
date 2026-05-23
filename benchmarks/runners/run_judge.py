"""Judging pass for multi-model benchmark.

Loads generation JSONL from run_generate.py and runs two judge passes:
  1. Citation support: YES/PARTIALLY/NO per claim-citation pair
  2. Answer quality: faithfulness 1-5, completeness 1-5, gaps

Source texts come from the generation JSONL (no re-retrieval needed), so the
judge model can differ completely from the generator model.

Output: the input JSONL augmented with judge_* fields, one record per line.

Run:
    python -m benchmarks.runners.run_judge --answers gen_8b.jsonl
    python -m benchmarks.runners.run_judge --answers gen_8b.jsonl --judge-model Qwen3.6-35B-A3B
    python -m benchmarks.runners.run_judge --answers gen_8b.jsonl --out judged_8b.jsonl
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

import config

_REPORTS_DIR = Path("benchmarks/reports/generations")

_CIT_RE = re.compile(r"\[([1-9]\d?)\]")
_SOURCES_SECTION_RE = re.compile(
    r"\n[*_-]*\s*\n?\s*(?:sources? cited|sources?|references?)[\s:*_-]*\n",
    re.IGNORECASE,
)
_VERDICT_RE = re.compile(r"\b(YES|PARTIALLY|NO)\b", re.IGNORECASE)

_JUDGE_CITATION_SYSTEM = (
    "You are a legal citation verifier. "
    "You will be given a claim from a legal research answer and the source passage it cites. "
    "Respond with exactly one word on the first line: YES, PARTIALLY, or NO. "
    "Then one sentence of explanation.\n\n"
    "YES = the passage directly supports the claim.\n"
    "PARTIALLY = the passage is related but only partially supports the claim.\n"
    "NO = the passage does not support the claim (or contradicts it)."
)

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
# Claim extraction (same logic as run_citation_support)
# ---------------------------------------------------------------------------

def _strip_sources_section(text: str) -> str:
    m = _SOURCES_SECTION_RE.search(text)
    return text[: m.start()] if m else text


def _extract_claim_pairs(answer: str) -> list[tuple[str, int]]:
    body = _strip_sources_section(answer)
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
        claim = _CIT_RE.sub("", line).strip(" .*-").strip()
        if len(claim) < 15:
            continue
        for idx in cits:
            if idx not in seen_idx:
                seen_idx.add(idx)
                pairs.append((claim, idx))
    return pairs


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

async def _llm_post(base_url: str, payload: dict) -> str:
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
                resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (httpx.RemoteProtocolError, httpx.TimeoutException,
                httpx.ConnectError, httpx.HTTPStatusError) as e:
            if attempt == 2:
                raise
            wait = 20 * (attempt + 1)
            print(f"  [retry {attempt + 1}/2 after {wait}s: {type(e).__name__}]")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def _judge_citation(
    claim: str, source_text: str, model: str, base_url: str
) -> tuple[str, str, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_CITATION_SYSTEM},
            {"role": "user", "content": f"Claim: {claim}\n\nSource passage:\n{source_text[:800]}"},
        ],
        "max_tokens": 80,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    text = await _llm_post(base_url, payload)
    latency_ms = (time.monotonic() - t0) * 1000
    m = _VERDICT_RE.search(text)
    verdict = m.group(1).upper() if m else "UNKNOWN"
    reason = text.split("\n", 1)[1].strip() if "\n" in text else text
    return verdict, reason, latency_ms


def _source_header(sources: list[dict]) -> str:
    return "\n".join(
        f"  [{i+1}] {s.get('title', 'Unknown')} | {s.get('court_name', '')} "
        f"| {s.get('date', '')} | {s.get('url', '')}"
        for i, s in enumerate(sources)
    )


async def _judge_quality(
    question: str, context_block: str, sources: list[dict], answer: str,
    model: str, base_url: str,
) -> tuple[int, int, str, float]:
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
        "model": model,
        "messages": [
            {"role": "system", "content": _QUALITY_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    text = await _llm_post(base_url, payload)
    latency_ms = (time.monotonic() - t0) * 1000
    faith = _parse_score(text, "FAITHFULNESS")
    complete = _parse_score(text, "COMPLETENESS")
    gaps = _parse_gaps(text)
    return faith, complete, gaps, latency_ms


async def _judge_noctx(
    question: str, context_block: str, sources: list[dict], answer: str,
    model: str, base_url: str,
) -> tuple[str, str, float]:
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
        "model": model,
        "messages": [
            {"role": "system", "content": _NOCTX_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 120,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    text = await _llm_post(base_url, payload)
    latency_ms = (time.monotonic() - t0) * 1000
    justified = "YES" if re.search(r"JUSTIFIED\s*:\s*YES", text, re.IGNORECASE) else "NO"
    reason_m = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE)
    reason = reason_m.group(1).strip() if reason_m else text
    return justified, reason, latency_ms


def _parse_score(text: str, label: str) -> int:
    m = re.search(rf"{label}\s*:\s*([1-5])", text, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _parse_gaps(text: str) -> str:
    m = re.search(r"GAPS\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip()[:400] if m else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    answers_path = Path(args.answers)
    records = [
        json.loads(line)
        for line in answers_path.read_text().splitlines()
        if line.strip()
    ]

    judge_model = args.judge_model or config.LLM_MODEL
    judge_base_url = args.judge_base_url or config.LLM_BASE_URL

    try:
        async with httpx.AsyncClient(base_url=judge_base_url, timeout=10) as client:
            resp = await client.get("/models")
            resp.raise_for_status()
        models_data = resp.json().get("data", [])
        judge_server_id = models_data[0]["id"] if models_data else judge_model
    except Exception:
        judge_server_id = judge_model

    out_path = Path(args.out) if args.out else (
        _REPORTS_DIR / f"judged_{answers_path.stem}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Judging pass")
    print(f"  answers:      {answers_path} ({len(records)} records)")
    print(f"  judge_model:  {judge_model} (server reports: {judge_server_id})")
    print(f"  judge_url:    {judge_base_url}")
    print(f"  out:          {out_path}")
    print()

    with out_path.open("w") as fout:
        for ri, rec in enumerate(records, 1):
            qid = rec["query_id"]
            query = rec["query"]
            answer = rec["answer"]
            sources = rec["sources"]
            context_block = rec["context_block"]
            no_ctx = rec.get("no_context_flag", 0)

            print(f"[{ri}/{len(records)}] {qid}")
            print(f"  Q: {query[:80]}")

            # Citation judging
            pairs = _extract_claim_pairs(answer)
            cit_results = []
            yes_n = partial_n = no_n = 0

            for claim, idx in pairs:
                if idx < 1 or idx > len(sources):
                    continue
                source_text = sources[idx - 1].get("text", "")
                source_case_id = sources[idx - 1].get("case_id", "")
                verdict, reason, _ = await _judge_citation(
                    claim, source_text, judge_model, judge_base_url
                )
                mark = {"YES": "+", "PARTIALLY": "~", "NO": "!"}.get(verdict, "?")
                print(f"  [{idx}] {mark} {verdict:9s}  {claim[:55]}")
                cit_results.append({
                    "claim": claim,
                    "citation_idx": idx,
                    "source_case_id": source_case_id,
                    "verdict": verdict,
                    "reason": reason,
                })
                if verdict == "YES":
                    yes_n += 1
                elif verdict == "PARTIALLY":
                    partial_n += 1
                else:
                    no_n += 1

            total_pairs = yes_n + partial_n + no_n
            cit_faithful = yes_n / total_pairs if total_pairs else 0.0

            # Quality judging
            faith, complete, gaps, qlat = await _judge_quality(
                query, context_block, sources, answer, judge_model, judge_base_url
            )
            faith_s = str(faith) if faith else "?"
            comp_s = str(complete) if complete else "?"
            print(f"  faithfulness={faith_s}/5  completeness={comp_s}/5  lat={qlat/1000:.1f}s")
            if gaps and gaps.lower() != "none":
                print(f"  gaps: {gaps[:100]}")

            # No-context check
            noctx_justified = "N/A"
            noctx_reason = ""
            if no_ctx:
                print(f"  [no_context_flag=1 - checking if justified]")
                noctx_justified, noctx_reason, _ = await _judge_noctx(
                    query, context_block, sources, answer, judge_model, judge_base_url
                )
                print(f"  justified={noctx_justified}: {noctx_reason[:80]}")

            augmented = {
                **rec,
                "judge_model": judge_model,
                "judge_server_id": judge_server_id,
                "judge_citation": cit_results,
                "citation_pairs_total": total_pairs,
                "citation_faithful_rate": round(cit_faithful, 4),
                "citation_yes": yes_n,
                "citation_partially": partial_n,
                "citation_no": no_n,
                "faithfulness": faith,
                "completeness": complete,
                "gaps": gaps,
                "noctx_justified": noctx_justified,
                "noctx_reason": noctx_reason,
            }
            fout.write(json.dumps(augmented) + "\n")
            fout.flush()

    # Summary from written file
    judged = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    valid_q = [r for r in judged if r.get("faithfulness", 0) > 0]
    avg_faith = sum(r["faithfulness"] for r in valid_q) / len(valid_q) if valid_q else 0.0
    avg_comp = sum(r["completeness"] for r in valid_q) / len(valid_q) if valid_q else 0.0
    total_cit = sum(r.get("citation_pairs_total", 0) for r in judged)
    total_yes = sum(r.get("citation_yes", 0) for r in judged)
    cit_rate = total_yes / total_cit if total_cit else 0.0
    noctx_n = sum(1 for r in judged if r.get("no_context_flag"))
    noctx_unjust = sum(1 for r in judged if r.get("noctx_justified") == "NO")

    print(f"\n--- Judge Summary ({judge_server_id}) ---")
    print(f"  Faithfulness mean: {avg_faith:.2f} / 5")
    print(f"  Completeness mean: {avg_comp:.2f} / 5")
    print(f"  Citation faithful: {total_yes}/{total_cit} = {cit_rate:.2f}")
    if noctx_n:
        print(f"  No-ctx queries: {noctx_n}, unjustified: {noctx_unjust}")
    print(f"\n  -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judging pass for multi-model benchmark")
    parser.add_argument("--answers", required=True,
                        help="JSONL file from run_generate.py")
    parser.add_argument("--judge-model", default=None,
                        help="Judge model name (default: config.LLM_MODEL)")
    parser.add_argument("--judge-base-url", default=None,
                        help="Judge server base URL (default: config.LLM_BASE_URL)")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path (default: judged_<stem>.jsonl in generations/)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
