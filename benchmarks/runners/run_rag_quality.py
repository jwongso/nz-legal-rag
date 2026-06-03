"""
RAG quality benchmark for the tenancy app.

Uses /ask/stream so the live RTA legislation anchor is included in generation.
Saves results to benchmarks/rag_quality_<slug>.json.

Usage:
    python -m benchmarks.runners.run_rag_quality --model Qwen3-8B-Q5_K_M
    python -m benchmarks.runners.run_rag_quality --model Negentropy-9B-Q5_K_M --out benchmarks/rag_quality_negentropy_9b.json
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

BASE_URL = "http://127.0.0.1:8081"

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


def _get_token() -> str:
    with urllib.request.urlopen(f"{BASE_URL}/token") as r:
        return json.loads(r.read())["token"]


def _stream_ask(question: str, token: str) -> dict:
    """Submit question to /ask/stream, collect SSE events, return result dict."""
    body = json.dumps({"question": question, "strategy": "vector"}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/ask/stream",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": token,
            "X-No-Log": "1",
        },
    )
    t0 = time.monotonic()
    answer_tokens: list[str] = []
    sources: list[dict] = []
    legislation: list[dict] = []
    verification: list[dict] = []
    confidence: dict = {}
    error: str = ""

    with urllib.request.urlopen(req, timeout=120) as resp:
        buf = b""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                raw, buf = buf.split(b"\n\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                try:
                    ev = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                t = ev.get("type", "")
                if t == "sources":
                    sources = ev.get("sources", [])
                    legislation = ev.get("legislation", [])
                elif t == "confidence":
                    confidence = {k: ev[k] for k in ("level", "chunks", "message") if k in ev}
                elif t == "token":
                    answer_tokens.append(ev.get("text", ""))
                elif t == "verification":
                    verification = ev.get("sections", [])
                elif t == "error":
                    error = ev.get("message", "unknown error")

    elapsed = time.monotonic() - t0
    answer = "".join(answer_tokens)
    # Strip trailing Sources: block if present
    idx = answer.rfind("\n\nSources:")
    if idx != -1:
        answer = answer[:idx].strip()

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "legislation": legislation,
        "verification": verification,
        "confidence": confidence,
        "elapsed_s": round(elapsed, 2),
        "error": error,
    }


def run(model_slug: str, out_path: Path) -> None:
    print(f"Benchmark: {model_slug}")
    print(f"Output: {out_path}")
    api_token = _get_token()

    results: dict[str, dict] = {}
    for key, question in QUESTIONS.items():
        print(f"  {key} ...", end=" ", flush=True)
        try:
            r = _stream_ask(question, api_token)
        except Exception as exc:
            r = {"question": question, "answer": "", "sources": [], "error": str(exc), "elapsed_s": 0}
        results[key] = r
        status = "ERROR" if r.get("error") else f"{r['elapsed_s']}s"
        print(status)

    payload = {
        "model": model_slug,
        "date": str(date.today()),
        "strategy": "vector+live_anchor",
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model slug for the result file")
    parser.add_argument("--out", default=None, help="Output path (default: auto from model slug)")
    args = parser.parse_args()

    slug = args.model.lower().replace(" ", "_").replace("-", "_")
    out = Path(args.out) if args.out else Path(f"benchmarks/rag_quality_{slug}.json")
    run(args.model, out)
