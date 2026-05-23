"""Generate benchmark visualization charts for README / BENCHMARK.md.

Produces four PNG files in benchmarks/reports/figures/:

  fig1_embed_retrieval.png   - Embedding shootout: H@5(gold), H@5(rel), MRR
  fig2_embed_answer.png      - Answer quality by task type: nomic vs Qwen3-Embedding
  fig3_quant_sweep.png       - Quantization sweep: faith/compl/cit with fixed judge
  fig4_retrieval_vs_answer.png - Scatter: retrieval MRR vs answer faithfulness

Run:
    python -m benchmarks.runners.plot_benchmarks
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_REPORTS = Path("benchmarks/reports")
_OUT = Path("benchmarks/figures")
_OUT.mkdir(exist_ok=True)

# Color palette - accessible, works in print
C_NOMIC    = "#4C72B0"  # blue
C_BGE      = "#DD8452"  # orange
C_E5       = "#55A868"  # green
C_QWEN_EMB = "#C44E52"  # red

C_Q4 = "#4C72B0"
C_Q5 = "#55A868"
C_Q6 = "#DD8452"

EMBED_COLORS = [C_NOMIC, C_BGE, C_E5, C_QWEN_EMB]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_embed_retrieval():
    """Load per-query retrieval scores for each embedder, averaged over both pipelines."""
    fnames = {
        "nomic":  "embed_nomic.json",
        "bge-m3": "embed_bge_m3.json",
        "e5-large": "embed_e5_large.json",
        "Qwen3-Emb": "embed_qwen3_06b.json",
    }
    rows = []
    for label, fname in fnames.items():
        d = json.loads((_REPORTS / fname).read_text())
        results = d["results"]
        pipelines = d["pipelines"]
        # average metric across both pipelines per query, then mean across queries
        def _mean_metric(key):
            per_q = [
                statistics.mean(r["scores"][p][key] for p in pipelines)
                for r in results
            ]
            return statistics.mean(per_q)
        rows.append({
            "label": label,
            "H@5(gold)": _mean_metric("hit_gold_at_5"),
            "H@5(rel)":  _mean_metric("hit_rel_at_5"),
            "MRR":       _mean_metric("mrr"),
        })
    return rows


def _load_answer_by_tasktype():
    """Load per-query answer quality grouped by task type for nomic and Qwen3-Embedding."""
    out = {}
    for label, fname in [("nomic", "judged_q5_nomic.jsonl"),
                          ("Qwen3-Emb", "judged_q5_qwen3emb.jsonl")]:
        records = [
            json.loads(l)
            for l in (_REPORTS / "generations" / fname).read_text().splitlines()
            if l.strip()
        ]
        by_type = {}
        for r in records:
            tt = r.get("task_type", "unknown")
            by_type.setdefault(tt, []).append(r)
        per_type = {}
        for tt, recs in by_type.items():
            valid = [r for r in recs if r.get("faithfulness", 0) > 0]
            per_type[tt] = {
                "faith": statistics.mean(r["faithfulness"] for r in valid) if valid else 0,
                "compl": statistics.mean(r["completeness"] for r in valid) if valid else 0,
                "n": len(valid),
            }
        out[label] = per_type
    return out


def _load_quant_sweep():
    """Load quant sweep results (Q4/Q5/Q6 judged by fixed Q4 judge)."""
    # Q4 judged itself; Q5 and Q6 judged by Q4
    quants = [
        ("Q4_K_M", "judged_q4.jsonl"),
        ("Q5_K_M", "judged_q5_by_q4.jsonl"),
        ("Q6_K",   "judged_q6_by_q4.jsonl"),
    ]
    rows = []
    for label, fname in quants:
        records = [
            json.loads(l)
            for l in (_REPORTS / "generations" / fname).read_text().splitlines()
            if l.strip()
        ]
        valid = [r for r in records if r.get("faithfulness", 0) > 0]
        total_cit = sum(r.get("citation_pairs_total", 0) for r in records)
        total_yes = sum(r.get("citation_yes", 0) for r in records)
        rows.append({
            "label": label,
            "faith": statistics.mean(r["faithfulness"] for r in valid) if valid else 0,
            "compl": statistics.mean(r["completeness"] for r in valid) if valid else 0,
            "cit_yes": total_yes / total_cit if total_cit else 0,
        })
    return rows


# ---------------------------------------------------------------------------
# Figure 1: Embedding shootout retrieval metrics
# ---------------------------------------------------------------------------

def fig1_embed_retrieval(rows):
    metrics = ["H@5(gold)", "H@5(rel)", "MRR"]
    n_metrics = len(metrics)
    n_models = len(rows)
    width = 0.18
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (row, color) in enumerate(zip(rows, EMBED_COLORS)):
        vals = [row[m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=row["label"], color=color,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Embedding Model Shootout - Retrieval Metrics\n(30 gold queries, averaged over oracle + planner pipelines)",
                 fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)

    # annotate the key insight
    ax.annotate("Qwen3-Emb: best gold precision\n+80% vs nomic on H@5(gold)",
                xy=(x[0] + 1.5 * width, rows[3]["H@5(gold)"] + 0.02),
                xytext=(x[0] + 2.5 * width, 0.75),
                fontsize=9, color=C_QWEN_EMB,
                arrowprops=dict(arrowstyle="->", color=C_QWEN_EMB, lw=1.2))

    fig.tight_layout()
    out = _OUT / "fig1_embed_retrieval.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Answer quality by task type
# ---------------------------------------------------------------------------

def fig2_embed_answer(data):
    task_types = ["general", "statute", "synthesis"]
    labels = list(data.keys())   # ["nomic", "Qwen3-Emb"]
    colors = [C_NOMIC, C_QWEN_EMB]
    metrics = [("faith", "Faithfulness"), ("compl", "Completeness")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        x = np.arange(len(task_types))
        width = 0.32
        for i, (label, color) in enumerate(zip(labels, colors)):
            vals = [data[label].get(tt, {}).get(metric_key, 0) for tt in task_types]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=color,
                          edgecolor="white", linewidth=0.8, alpha=0.9)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in task_types], fontsize=11)
        ax.set_title(metric_label, fontsize=12, pad=8)
        ax.set_ylim(0, 5.4)
        ax.set_ylabel("Score (1-5)", fontsize=11)
        ax.yaxis.set_tick_params(labelleft=True)

    axes[0].legend(loc="upper left", fontsize=10, framealpha=0.8)

    fig.suptitle("Answer Quality: nomic vs Qwen3-Embedding-0.6B\n"
                 "(same Q5_K_M generator + judge, 20 questions)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = _OUT / "fig2_embed_answer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Quantization sweep
# ---------------------------------------------------------------------------

def fig3_quant_sweep(rows):
    quant_labels = [r["label"] for r in rows]
    colors = [C_Q4, C_Q5, C_Q6]
    metrics = [("faith", "Faithfulness (1-5)"), ("compl", "Completeness (1-5)"), ("cit_yes", "Citation YES rate")]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        scale = 5.0 if key != "cit_yes" else 1.0
        bars = ax.bar(quant_labels, vals, color=colors, edgecolor="white", linewidth=0.8,
                      width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * scale,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(0, scale * 1.15)
        if key == "cit_yes":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # highlight Q5 as selected
    for ax in axes:
        ax.get_children()[1].set_edgecolor("black")  # Q5 bar index 1
        ax.get_children()[1].set_linewidth(2.5)

    fig.suptitle("Quantization Sweep: Q4 / Q5 / Q6  (fixed Q4 judge)\n"
                 "Q5_K_M selected for production  [bold border]",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = _OUT / "fig3_quant_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Retrieval MRR vs Answer Faithfulness scatter
# ---------------------------------------------------------------------------

def fig4_scatter(embed_rows, answer_data):
    # per-embedder: MRR from retrieval, faithfulness from answer (overall)
    embed_faith = {}
    for label, fname in [("nomic", "judged_q5_nomic.jsonl"),
                          ("Qwen3-Emb", "judged_q5_qwen3emb.jsonl")]:
        records = [
            json.loads(l)
            for l in (_REPORTS / "generations" / fname).read_text().splitlines()
            if l.strip()
        ]
        valid = [r for r in records if r.get("faithfulness", 0) > 0]
        embed_faith[label] = statistics.mean(r["faithfulness"] for r in valid) if valid else 0

    # nomic and Qwen3-Emb are at index 0 and 3 in embed_rows
    points = [
        ("nomic", embed_rows[0]["MRR"], embed_faith["nomic"], C_NOMIC),
        ("bge-m3", embed_rows[1]["MRR"], None, C_BGE),
        ("e5-large", embed_rows[2]["MRR"], None, C_E5),
        ("Qwen3-Emb", embed_rows[3]["MRR"], embed_faith["Qwen3-Emb"], C_QWEN_EMB),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, mrr, faith, color in points:
        if faith is not None:
            ax.scatter(mrr, faith, s=200, color=color, zorder=3, edgecolors="white", linewidth=1.5)
            va = "bottom" if label == "nomic" else "top"
            offset = 0.01 if label == "nomic" else -0.01
            ax.annotate(label, (mrr, faith + offset), ha="center", va=va,
                        fontsize=11, fontweight="bold", color=color)
        else:
            # retrieval only, no answer quality run - show on MRR axis
            ax.axvline(mrr, color=color, linestyle=":", alpha=0.6, linewidth=1.5)
            ax.text(mrr + 0.003, 3.25, label, color=color, fontsize=9,
                    rotation=90, va="bottom")

    ax.set_xlabel("Retrieval MRR (higher = better gold document ranking)", fontsize=11)
    ax.set_ylabel("Answer Faithfulness mean (1-5)", fontsize=11)
    ax.set_title("Retrieval Precision vs Answer Faithfulness\n"
                 "Dotted lines = retrieval-only benchmark (no answer run)",
                 fontsize=11, pad=10)

    # draw a trend arrow
    x1, y1 = embed_rows[0]["MRR"], embed_faith["nomic"]
    x2, y2 = embed_rows[3]["MRR"], embed_faith["Qwen3-Emb"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5, linestyle="dashed"))

    fig.tight_layout()
    out = _OUT / "fig4_retrieval_vs_answer.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    embed_rows  = _load_embed_retrieval()
    answer_data = _load_answer_by_tasktype()
    quant_rows  = _load_quant_sweep()

    print("Generating figures...")
    fig1_embed_retrieval(embed_rows)
    fig2_embed_answer(answer_data)
    fig3_quant_sweep(quant_rows)
    fig4_scatter(embed_rows, answer_data)

    print(f"\nAll figures saved to {_OUT}/")


if __name__ == "__main__":
    main()
