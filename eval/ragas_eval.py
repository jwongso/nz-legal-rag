"""
RAGAS evaluation of the RAG pipeline against the NZ legal Q&A benchmark.

Usage:
    python -m eval.ragas_eval --questions eval/questions.jsonl --output eval/results.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from rag.pipeline import RAGPipeline


async def collect_responses(questions_path: Path, pipeline: RAGPipeline) -> list[dict]:
    rows = []
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line.strip())
            question = item["question"]
            ground_truth = item.get("ground_truth", "")

            response = await pipeline.ask(question, top_k=5)

            rows.append({
                "question": question,
                "answer": response.answer,
                "contexts": response.context_texts if response.context_texts else [],
                "ground_truth": ground_truth,
            })
            print(f"  Q: {question[:60]}...")
    return rows


def run_eval(questions_path: Path, output_path: Path) -> None:
    pipeline = RAGPipeline()
    rows = asyncio.run(collect_responses(questions_path, pipeline))

    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }

    print("\n--- RAGAS Results ---")
    for metric, score in scores.items():
        status = "PASS" if score >= 0.70 else "FAIL"
        print(f"  {metric:<25} {score:.3f}  [{status}]")

    output_path.write_text(json.dumps({"scores": scores, "rows": rows}, indent=2))
    print(f"\nFull results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/results.json"))
    args = parser.parse_args()
    run_eval(args.questions, args.output)


if __name__ == "__main__":
    main()
