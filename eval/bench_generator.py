"""Generator LLM benchmark: TTFT, tokens/sec, citation correctness, RAM/VRAM.

Measures the speed and citation discipline of the local LLM without touching
the retrieval stack. Uses a fixed realistic NZ legal context so results are
reproducible across runs and comparable across model swaps.

Metrics:
  ttft          Time to first token (streaming, seconds)
  tps           Output tokens per second (estimated from chars / 4)
  citation_ok   Fraction of [N] references that point to a valid source number
  total_s       Wall time for the full generation

Hardware snapshot (GPU/RAM) is recorded before and after the run.

Run:
    python -m eval.bench_generator
    python -m eval.bench_generator --quick
    python -m eval.bench_generator --llm-url http://localhost:8080/v1 --model qwen3
"""

import argparse
import asyncio
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil

import config

_RESULTS_DIR = Path("eval/bench_results")

_SYSTEM_PROMPT = (
    "You are a legal research assistant for New Zealand law. "
    "Answer only from the provided context. "
    "Cite sources with [N] notation matching the source index."
)

# Fixed two-source context so citation checks are deterministic
_CONTEXT = """\
[1] Section 54 of the Residential Tenancies Act 1986 provides that a landlord \
must give the tenant at least 24 hours written notice before entering the premises, \
except in cases of emergency. Failure to comply may result in exemplary damages.

[2] The Employment Relations Act 2000, section 103A, requires an employer to \
follow a fair and reasonable process before dismissing an employee. The Employment \
Relations Authority may order reinstatement or compensation for unjustified dismissal."""

_N_SOURCES = 2


def _gpu_stats() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            return {
                "vram_used_mb": int(parts[0]),
                "vram_total_mb": int(parts[1]),
                "gpu_util_pct": int(parts[2]),
            }
    except Exception:
        pass
    return None


def _ram_stats() -> dict:
    mem = psutil.virtual_memory()
    return {
        "ram_used_gb": round(mem.used / 1024 ** 3, 2),
        "ram_total_gb": round(mem.total / 1024 ** 3, 2),
        "ram_pct": mem.percent,
    }


def _citation_correctness(answer: str, n_sources: int) -> float:
    refs = re.findall(r'\[(\d+)\]', answer)
    if not refs:
        return 1.0
    valid = sum(1 for r in refs if 1 <= int(r) <= n_sources)
    return valid / len(refs)


async def _stream_ask(client: httpx.AsyncClient, question: str, model: str) -> dict:
    user_msg = (
        f"Source index:\n{_CONTEXT}\n\n"
        f"Question: {question}\n\n"
        "Answer from the context above. Use [N] citation notation."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": config.LLM_MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    ttft = None
    parts: list[str] = []
    completion_tokens = 0
    t0 = time.monotonic()

    try:
        async with client.stream("POST", "/chat/completions", json=payload, timeout=120) as resp:
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if "usage" in chunk:
                        completion_tokens = chunk["usage"].get("completion_tokens", 0)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        parts.append(delta)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except Exception as exc:
        return {"error": str(exc)}

    elapsed = time.monotonic() - t0
    answer = "".join(parts)
    out_tokens = completion_tokens if completion_tokens > 0 else max(1, len(answer) // 4)

    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(elapsed, 3),
        "output_chars": len(answer),
        "output_tokens_est": out_tokens,
        "tps": round(out_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        "citation_correctness": round(_citation_correctness(answer, _N_SOURCES), 3),
        "answer_preview": answer[:120].replace("\n", " "),
    }


async def run(questions_path: Path, llm_url: str, model: str, quick: bool) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(questions_path) as f:
        questions = [json.loads(l) for l in f if l.strip()]
    if quick:
        questions = questions[:3]

    print(f"Generator benchmark  model={model}  url={llm_url}")
    print(f"Questions: {len(questions)}")
    print()

    gpu_before = _gpu_stats()
    ram_before = _ram_stats()

    if gpu_before:
        print(f"  VRAM: {gpu_before['vram_used_mb']} / {gpu_before['vram_total_mb']} MB  "
              f"({gpu_before['gpu_util_pct']}% util)")
    print(f"  RAM:  {ram_before['ram_used_gb']} / {ram_before['ram_total_gb']} GB  "
          f"({ram_before['ram_pct']}%)")
    print()

    runs = []
    async with httpx.AsyncClient(base_url=llm_url) as client:
        for i, item in enumerate(questions, 1):
            q = item["question"]
            print(f"[{i}/{len(questions)}] {q[:70]}", end="  ", flush=True)
            r = await _stream_ask(client, q, model)
            runs.append({"question": q, **r})
            if "error" in r:
                print(f"ERROR: {r['error']}")
            else:
                print(
                    f"ttft={r['ttft_s']}s  tps={r['tps']}  "
                    f"cite={r['citation_correctness']:.0%}"
                )

    gpu_after = _gpu_stats()

    ok = [r for r in runs if "error" not in r]
    summary: dict = {"model": model, "questions_run": len(runs), "successful": len(ok)}
    if ok:
        ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
        tpss = [r["tps"] for r in ok]
        cites = [r["citation_correctness"] for r in ok]
        summary.update({
            "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
            "ttft_min_s":  round(min(ttfts), 3) if ttfts else None,
            "ttft_max_s":  round(max(ttfts), 3) if ttfts else None,
            "tps_mean":    round(sum(tpss) / len(tpss), 1),
            "tps_min":     round(min(tpss), 1),
            "tps_max":     round(max(tpss), 1),
            "citation_correctness_mean": round(sum(cites) / len(cites), 3),
            "gpu_before": gpu_before,
            "gpu_after":  gpu_after,
            "ram_before": ram_before,
        })

    print()
    print("--- Generator Benchmark Summary ---")
    if ok:
        print(f"  Questions:    {summary['questions_run']}  ({summary['successful']} ok)")
        if summary.get("ttft_mean_s") is not None:
            print(
                f"  TTFT:         mean={summary['ttft_mean_s']}s  "
                f"min={summary['ttft_min_s']}s  max={summary['ttft_max_s']}s"
            )
        print(
            f"  Tokens/sec:   mean={summary['tps_mean']}  "
            f"min={summary['tps_min']}  max={summary['tps_max']}"
        )
        print(f"  Citation ok:  {summary['citation_correctness_mean']:.1%}")
        if gpu_before and gpu_after:
            print(
                f"  VRAM:         {gpu_before['vram_used_mb']} MB before / "
                f"{gpu_after['vram_used_mb']} MB after"
            )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _RESULTS_DIR / f"generator_{ts}.json"
    out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=2))
    print(f"\nResults -> {out}")
    return {"summary": summary, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator LLM benchmark")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--llm-url", default=config.LLM_BASE_URL)
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--quick", action="store_true", help="Run only 3 questions")
    args = parser.parse_args()
    asyncio.run(run(args.questions, args.llm_url, args.model, args.quick))


if __name__ == "__main__":
    main()
