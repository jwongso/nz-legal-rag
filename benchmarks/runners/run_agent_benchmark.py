"""
Agentic RAG benchmark: models get web_search + fetch_url tools.

RAG context (tribunal decisions + live RTA anchor) is pre-included in the prompt.
Models can use tools to search the web or fetch specific URLs to verify or supplement
that context with current legislation, regulations, and official guidance.

Usage:
    python -m benchmarks.runners.run_agent_benchmark --model Qwen3-8B-Q5_K_M
    python -m benchmarks.runners.run_agent_benchmark --model Mistral-7B-Instruct-v0.3-Q5_K_M --out benchmarks/agent_mistral_7b.json
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
BASE_URL = "http://127.0.0.1:8081"
MAX_TOOL_ROUNDS = 6
FETCH_CHAR_LIMIT = 2500

QUESTIONS = {
    "Q1_no_cause_termination": (
        "My landlord gave me 90 days notice to vacate but didn't give any reason. "
        "The tenancy is periodic and started in 2022. Is this notice valid?"
    ),
    "Q2_fixed_term_expiry": (
        "My fixed term tenancy ended 3 weeks ago. My landlord says I have no rights "
        "because the contract expired. Neither of us signed a new agreement. Do I have to leave?"
    ),
    "Q3_retaliatory_notice": (
        "I complained to Tenancy Services about mould on 1 March. My landlord gave me a "
        "termination notice on 15 April citing sale of property. Is there anything I can do?"
    ),
    "Q4_rent_reduction_clock": (
        "My landlord reduced my rent from $600 to $550 per week in January by agreement. "
        "In August (7 months later) they want to increase it back to $600. Under the "
        "12-month rule, can they do this?"
    ),
    "Q5_meth_threshold": (
        "The landlord tested the property after I moved out and found methamphetamine "
        "contamination of 2.0 micrograms per 100cm2. They are claiming the full bond plus "
        "additional damages for remediation. Are they entitled to this?"
    ),
    "Q6_bond_topup": (
        "My rent increased from $500 to $600 per week last month. My landlord is now "
        "demanding I pay an extra $400 bond to bring it up to 4 weeks of the new rent. "
        "Is this legal?"
    ),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current NZ tenancy law information. "
                "Use this to find current thresholds, recent law changes, notice periods, "
                "or official guidance from tenancy.govt.nz, hud.govt.nz, or legislation.govt.nz."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'NZ meth contamination threshold 2026 regulations site:hud.govt.nz'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch the text content of a specific URL. "
                "Use for official NZ government pages such as tenancy.govt.nz, "
                "hud.govt.nz, or legislation.govt.nz."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a NZ residential tenancy law assistant.\n\n"
    "MANDATORY RULE: You MUST call web_search or fetch_url BEFORE giving your final answer. "
    "NZ tenancy law changes frequently - thresholds, notice periods, and sections are amended regularly. "
    "Never answer from memory alone. Always verify using your tools first.\n\n"
    "Suggested searches:\n"
    "- web_search the specific legal issue (e.g. 'NZ retaliatory notice tenancy 2025 site:tenancy.govt.nz')\n"
    "- web_search for recent amendments (e.g. 'Residential Tenancies Amendment Act 2024 NZ')\n"
    "- fetch_url a specific official page if you know the URL\n\n"
    "You have been given relevant NZ Tenancy Tribunal decisions as background context. "
    "Use your tools to verify and supplement that context with current official sources.\n\n"
    "After searching, give a clear, direct answer citing the relevant law sections."
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_token() -> str:
    with urllib.request.urlopen(f"{BASE_URL}/token") as r:
        return json.loads(r.read())["token"]


def _get_rag_context(question: str, token: str) -> dict:
    body = json.dumps({"question": question, "strategy": "vector"}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/retrieve",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": token},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _web_search(query: str, token: str = "") -> str:
    """Search via the tenancy app's /search endpoint (Playwright Firefox, undetected)."""
    try:
        body = json.dumps({"query": query, "max_results": 5}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/search",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": token},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r['title']}**\n{r['url']}\n{r['body']}\n")
        return "\n".join(lines)
    except Exception as exc:
        return f"Search error: {exc}"


def _fetch_url(url: str) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:FETCH_CHAR_LIMIT]
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} error fetching {url}. Try a different URL or use web_search instead."
    except Exception as exc:
        return f"Fetch error: {exc}"


def _execute_tool(name: str, args: dict, token: str = "") -> str:
    if name == "web_search":
        return _web_search(args.get("query", ""), token=token)
    if name == "fetch_url":
        return _fetch_url(args.get("url", ""))
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _chat(messages: list, max_tokens: int = 2500, force_tool: bool = False) -> dict:
    payload: dict = {
        "model": "local",
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "thinking": {"type": "disabled"},  # suppress reasoning tokens for Qwen3
    }
    if force_tool:
        payload["tool_choice"] = "required"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        LLAMA_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def _format_context(rag: dict) -> str:
    """Format RAG context into a readable block for the prompt."""
    lines = []
    anchor = rag.get("anchor", "")
    if anchor:
        lines.append(anchor)
        lines.append("")
    ctx = rag.get("context_texts", [])
    sources = rag.get("sources", [])
    if ctx:
        lines.append("Relevant NZ Tenancy Tribunal decisions:")
        for i, (text, src) in enumerate(zip(ctx, sources), 1):
            citation = src.get("citation", f"Source {i}")
            lines.append(f"\n[S{i}] {citation}\n{text}")
    return "\n".join(lines)


def _run_agent(question: str, rag: dict, token: str = "", verbose: bool = False) -> dict:
    context_block = _format_context(rag)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{context_block}\n\n"
                f"---\n\nQuestion: {question}"
            ),
        },
    ]

    tool_log: list[dict] = []
    rounds = 0

    for round_idx in range(MAX_TOOL_ROUNDS):
        rounds += 1
        # Force a tool call on the first round to guarantee at least one search
        force = (round_idx == 0)
        try:
            resp = _chat(messages, max_tokens=3500, force_tool=force)
        except Exception as exc:
            return {"answer": "", "error": str(exc), "tool_log": tool_log, "rounds": rounds}

        choice = resp["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason", "")

        if finish == "tool_calls" or msg.get("tool_calls"):
            messages.append(msg)
            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                t_tool = time.monotonic()
                result = _execute_tool(fn["name"], args, token=token)
                elapsed_ms = round((time.monotonic() - t_tool) * 1000)
                tool_log.append({
                    "tool": fn["name"],
                    "args": args,
                    "elapsed_ms": elapsed_ms,
                    "result_preview": result[:200],
                })
                if verbose:
                    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    preview = result[:120].replace("\n", " ")
                    print(f"      [{elapsed_ms}ms] {fn['name']}({arg_str})")
                    print(f"             -> {preview}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })
        else:
            return {
                "answer": msg.get("content", ""),
                "error": "",
                "tool_log": tool_log,
                "rounds": rounds,
                "finish_reason": finish,
            }

    return {
        "answer": msg.get("content", ""),
        "error": "max_tool_rounds_exceeded",
        "tool_log": tool_log,
        "rounds": rounds,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(model_slug: str, out_path: Path, verbose: bool = False) -> None:
    print(f"Agent benchmark: {model_slug}")
    print(f"Output: {out_path}")
    token = _get_token()

    results: dict[str, dict] = {}
    for key, question in QUESTIONS.items():
        print(f"  {key} ...", end="\n" if verbose else " ", flush=True)
        t0 = time.monotonic()
        try:
            rag = _get_rag_context(question, token)
            result = _run_agent(question, rag, token=token, verbose=verbose)
            result["question"] = question
            result["elapsed_s"] = round(time.monotonic() - t0, 2)
            tools_used = [(t["tool"], t.get("elapsed_ms")) for t in result.get("tool_log", [])]
            tool_summary = ", ".join(f"{t}({ms}ms)" if ms else t for t, ms in tools_used)
            prefix = "  -> " if verbose else ""
            print(f"{prefix}{result['elapsed_s']}s  rounds={result['rounds']}  tools=[{tool_summary}]")
        except Exception as exc:
            result = {
                "question": question,
                "answer": "",
                "error": str(exc),
                "tool_log": [],
                "rounds": 0,
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
            print(f"ERROR: {exc}")
        results[key] = result

    payload = {
        "model": model_slug,
        "date": str(date.today()),
        "mode": "agentic",
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model slug for the result file")
    parser.add_argument("--out", default=None, help="Output path (default: auto from slug)")
    parser.add_argument("--verbose", action="store_true", help="Print each tool call with args and duration")
    args = parser.parse_args()

    slug = args.model.lower().replace(" ", "_").replace("-", "_")
    out = Path(args.out) if args.out else Path(f"benchmarks/agent_{slug}.json")
    run(args.model, out, verbose=args.verbose)
