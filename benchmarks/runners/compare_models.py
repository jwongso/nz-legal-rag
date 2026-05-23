"""Model comparison table generator for multi-model benchmark.

Loads one or more judged JSONL files (from run_judge.py) and produces
a Markdown comparison table and JSON summary.

Each file is identified by its generator_model / server_model_id fields.
Records must have judge scores added by run_judge.py.

Run:
    python -m benchmarks.runners.compare_models benchmarks/reports/generations/judged_*.jsonl
    python -m benchmarks.runners.compare_models --out report.md judged_8b.jsonl judged_35b.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path

_REPORTS_DIR = Path("benchmarks/reports")


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _model_label(records: list[dict]) -> str:
    sid = records[0].get("server_model_id", "")
    gm = records[0].get("generator_model", "")
    base = sid if sid and sid != gm else gm
    em = records[0].get("embed_model", "")
    # Shorten embed model to last path component if present
    em_short = em.split("/")[-1] if em else ""
    if em_short:
        return f"{base} + {em_short}"
    return base


def _aggregate(records: list[dict]) -> dict:
    n = len(records)

    valid_q = [r for r in records if r.get("faithfulness", 0) > 0]
    avg_faith = statistics.mean(r["faithfulness"] for r in valid_q) if valid_q else 0.0
    avg_comp = statistics.mean(r["completeness"] for r in valid_q) if valid_q else 0.0

    total_cit = sum(r.get("citation_pairs_total", 0) for r in records)
    total_yes = sum(r.get("citation_yes", 0) for r in records)
    cit_rate = total_yes / total_cit if total_cit else 0.0

    noctx_n = sum(1 for r in records if r.get("no_context_flag"))
    noctx_unjust = sum(1 for r in records if r.get("noctx_justified") == "NO")

    timings = [r.get("timing", {}) for r in records]
    ttfts = sorted(t["ttft_ms"] for t in timings if t.get("ttft_ms"))
    tok_rates = [t["tokens_per_sec"] for t in timings if t.get("tokens_per_sec")]
    total_ms_vals = [t["total_ms"] for t in timings if t.get("total_ms")]

    ttft_p50 = ttfts[len(ttfts) // 2] if ttfts else 0.0
    tok_mean = statistics.mean(tok_rates) if tok_rates else 0.0
    latency_mean = statistics.mean(total_ms_vals) / 1000 if total_ms_vals else 0.0

    pack_mode = records[0].get("context_pack_mode", "")
    judge_model = records[0].get("judge_server_id", records[0].get("judge_model", ""))

    return {
        "n": n,
        "avg_faith": avg_faith,
        "avg_comp": avg_comp,
        "cit_faithful_rate": cit_rate,
        "total_cit_pairs": total_cit,
        "noctx_n": noctx_n,
        "noctx_unjust": noctx_unjust,
        "ttft_p50_ms": ttft_p50,
        "tok_per_sec": tok_mean,
        "latency_mean_s": latency_mean,
        "pack_mode": pack_mode,
        "judge_model": judge_model,
    }


def _per_tasktype(records: list[dict]) -> dict[str, dict]:
    by_type: dict[str, list[dict]] = {}
    for r in records:
        tt = r.get("task_type", "unknown")
        by_type.setdefault(tt, []).append(r)
    return {tt: _aggregate(recs) for tt, recs in sorted(by_type.items())}


def _make_table(entries: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "| Model | N | Faith | Compl | Cit YES | no_ctx | TTFT p50 | tok/s | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, agg in entries:
        lines.append(
            f"| {label} "
            f"| {agg['n']} "
            f"| {agg['avg_faith']:.2f} "
            f"| {agg['avg_comp']:.2f} "
            f"| {agg['cit_faithful_rate']:.2f} "
            f"| {agg['noctx_n']}/{agg['n']} "
            f"| {agg['ttft_p50_ms']:.0f} ms "
            f"| {agg['tok_per_sec']:.1f} "
            f"| {agg['latency_mean_s']:.1f} s |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare judged generation JSONL files")
    parser.add_argument("files", nargs="+", help="Judged JSONL files from run_judge.py")
    parser.add_argument("--out", default=None,
                        help="Output Markdown (default: benchmarks/reports/model_comparison.md)")
    parser.add_argument("--json-out", default=None,
                        help="Output JSON (default: benchmarks/reports/model_comparison.json)")
    args = parser.parse_args()

    models: list[tuple[str, dict, dict]] = []
    for fpath in args.files:
        records = _load(Path(fpath))
        if not records:
            print(f"Warning: {fpath} is empty, skipping")
            continue
        label = _model_label(records)
        agg = _aggregate(records)
        by_type = _per_tasktype(records)
        models.append((label, agg, by_type))
        print(
            f"  {label}: n={agg['n']}  "
            f"faith={agg['avg_faith']:.2f}  "
            f"compl={agg['avg_comp']:.2f}  "
            f"cit={agg['cit_faithful_rate']:.2f}  "
            f"ttft_p50={agg['ttft_p50_ms']:.0f}ms  "
            f"tok/s={agg['tok_per_sec']:.1f}"
        )

    if not models:
        print("No valid files found.")
        return

    judge_labels = list(dict.fromkeys(agg["judge_model"] for _, agg, _ in models))
    judge_str = ", ".join(judge_labels) if judge_labels else "unknown"

    all_task_types = sorted({tt for _, _, by_type in models for tt in by_type})

    lines = [
        "# Multi-Model Comparison",
        "",
        f"Judge: {judge_str}",
        "Retrieval pipeline: planner_filter_vector_legal",
        "",
        "## Overall",
        "",
    ]
    lines += _make_table([(label, agg) for label, agg, _ in models])
    lines.append("")

    if len(all_task_types) > 1:
        for tt in all_task_types:
            lines += [f"## By Task Type: {tt}", ""]
            tt_entries = [
                (label, by_type[tt])
                for label, _, by_type in models
                if tt in by_type
            ]
            if tt_entries:
                lines += _make_table(tt_entries)
            lines.append("")

    lines += [
        "## Column Notes",
        "",
        "- **Faith** = faithfulness mean (1-5, judge-rated, higher is better)",
        "- **Compl** = completeness mean (1-5, judge-rated, higher is better)",
        "- **Cit YES** = fraction of claim-citation pairs judged fully supported (higher is better)",
        "- **no_ctx** = queries where model said 'not enough context' (lower is better)",
        "- **TTFT p50** = median time-to-first-token in ms (lower is better)",
        "- **tok/s** = mean generation throughput (higher is better)",
        "- **Latency** = mean total generation time in seconds (lower is better)",
        "",
    ]

    out_path = Path(args.out) if args.out else _REPORTS_DIR / "model_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n  -> {out_path}")

    json_out = Path(args.json_out) if args.json_out else _REPORTS_DIR / "model_comparison.json"
    summary = [
        {"model": label, "aggregate": agg, "by_task_type": by_type}
        for label, agg, by_type in models
    ]
    json_out.write_text(json.dumps(summary, indent=2))
    print(f"  -> {json_out}")


if __name__ == "__main__":
    main()
