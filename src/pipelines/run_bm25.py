from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_yaml_config
from src.data.evidence import build_evidence_chunks, find_supported_gold_answer, parse_json_list
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils import ensure_dir, slugify, write_json


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    output_defaults = experiment_config.get("outputs", {})
    retrieval_defaults = experiment_config.get("retrieval", {})

    parser = argparse.ArgumentParser(description="Run a BM25 evidence-retrieval baseline.")
    parser.add_argument(
        "--benchmark",
        default=retrieval_defaults.get("benchmark_path", "data/benchmark/triviaqa_pilot_v1.csv"),
        help="Benchmark CSV path, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(retrieval_defaults.get("top_k", 5)),
        help="Number of passages to retrieve per question.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(retrieval_defaults.get("chunk_size", 120)),
        help="Evidence chunk size in words.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=int(retrieval_defaults.get("overlap", 30)),
        help="Overlapping words between adjacent chunks.",
    )
    parser.add_argument(
        "--max-chunks-per-question",
        type=int,
        default=int(retrieval_defaults.get("max_chunks_per_question", 200)),
        help="Maximum evidence chunks indexed per question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=retrieval_defaults.get("max_questions"),
        help="Optional limit on benchmark rows.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV path. If omitted, a deterministic BM25 run file is created.",
    )
    parser.add_argument(
        "--runs-dir",
        default=output_defaults.get("runs_dir", "results/runs"),
        help="Directory for generated retrieval artifacts.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def build_output_path(args: argparse.Namespace, benchmark_path: Path) -> Path:
    if args.output:
        return resolve_path(args.output)

    benchmark_name = slugify(benchmark_path.stem)
    filename = f"bm25__{benchmark_name}__top{args.top_k}.csv"
    return resolve_path(args.runs_dir) / filename


def main() -> None:
    args = parse_args()

    benchmark_path = resolve_path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    benchmark_df = pd.read_csv(benchmark_path)
    if args.limit is not None:
        benchmark_df = benchmark_df.head(args.limit).copy()

    output_path = build_output_path(args, benchmark_path)
    ensure_dir(output_path.parent)

    result_rows: list[dict[str, object]] = []
    for row in tqdm(
        benchmark_df.to_dict(orient="records"),
        total=len(benchmark_df),
        desc="Running BM25 retrieval",
    ):
        question = str(row.get("question") or "").strip()
        gold_answers = parse_json_list(row.get("gold_aliases_json"))
        evidence_chunks = build_evidence_chunks(
            row,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_chunks=args.max_chunks_per_question,
        )

        retrieved_records: list[dict[str, object]] = []
        supported_answer = ""
        supported_rank = None

        if evidence_chunks:
            corpus = [str(chunk["text"]) for chunk in evidence_chunks]
            retriever = BM25Retriever(corpus)
            retrieved = retriever.search(question, top_k=args.top_k)
            text_to_chunks = {str(chunk["text"]): chunk for chunk in evidence_chunks}

            for rank, (passage, score) in enumerate(retrieved, start=1):
                chunk_metadata = text_to_chunks.get(passage, {})
                matched_answer = find_supported_gold_answer(passage, gold_answers)
                if matched_answer and not supported_answer:
                    supported_answer = matched_answer
                    supported_rank = rank
                retrieved_records.append(
                    {
                        "rank": rank,
                        "score": round(float(score), 6),
                        "source": chunk_metadata.get("source", ""),
                        "context_index": chunk_metadata.get("context_index", ""),
                        "chunk_index": chunk_metadata.get("chunk_index", ""),
                        "matched_gold_answer": matched_answer,
                        "passage": passage,
                    }
                )

        result_rows.append(
            {
                "question_id": row.get("question_id"),
                "question": question,
                "gold_primary": row.get("gold_primary"),
                "gold_aliases_json": row.get("gold_aliases_json"),
                "evidence_chunk_count": len(evidence_chunks),
                "retrieved_passages_json": json.dumps(retrieved_records, ensure_ascii=False),
                "gold_supported_in_top_k": bool(supported_answer),
                "supported_gold_answer": supported_answer,
                "supported_rank": supported_rank,
                "retrieval_error": "" if evidence_chunks else "No evidence chunks available",
            }
        )

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".summary.json")
    valid_rows = results_df[results_df["retrieval_error"] == ""]
    summary = {
        "run_type": "bm25_retrieval",
        "benchmark_path": str(benchmark_path),
        "output_path": str(output_path),
        "row_count": int(len(results_df)),
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "max_chunks_per_question": args.max_chunks_per_question,
        "no_evidence_rows": int((results_df["retrieval_error"] != "").sum()),
        "mean_evidence_chunks": round(float(results_df["evidence_chunk_count"].mean()), 4),
        "gold_support_rate_top_k": round(float(results_df["gold_supported_in_top_k"].mean()), 6),
        "gold_support_rate_top_k_with_evidence": round(float(valid_rows["gold_supported_in_top_k"].mean()), 6)
        if len(valid_rows)
        else 0.0,
    }
    write_json(summary, summary_path)

    print(f"Wrote BM25 retrieval results to {output_path}")
    print(f"Wrote BM25 summary to {summary_path}")
    print(f"Gold support@{args.top_k}: {summary['gold_support_rate_top_k']:.3f}")


if __name__ == "__main__":
    main()
