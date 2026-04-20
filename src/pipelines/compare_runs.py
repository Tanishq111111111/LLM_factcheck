from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_yaml_config
from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    output_defaults = experiment_config.get("outputs", {})

    parser = argparse.ArgumentParser(description="Compare direct, BM25, and RAG pilot outputs.")
    parser.add_argument(
        "--direct-metrics",
        default="results/tables/direct-openai-gpt-5-4-mini-pilot10__metrics.csv",
        help="Direct baseline metrics CSV.",
    )
    parser.add_argument(
        "--rag-metrics",
        default="results/tables/rag-openai-gpt-5-4-mini-bm25-top5-pilot10__metrics.csv",
        help="RAG metrics CSV.",
    )
    parser.add_argument(
        "--bm25-results",
        default="results/runs/bm25__triviaqa_pilot_v1__top5.csv",
        help="BM25 retrieval results CSV.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional comparison CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional comparison summary JSON path.",
    )
    parser.add_argument(
        "--tables-dir",
        default=output_defaults.get("tables_dir", "results/tables"),
        help="Directory for comparison artifacts.",
    )
    parser.add_argument(
        "--review-output",
        default="",
        help="Optional manual review CSV path.",
    )
    parser.add_argument(
        "--reviews-dir",
        default=output_defaults.get("reviews_dir", "results/manual_reviews"),
        help="Directory for manual review artifacts.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_required_csv(path_text: str) -> pd.DataFrame:
    path = resolve_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def outcome_bucket(direct_label: str, rag_label: str) -> str:
    if direct_label == rag_label:
        return "unchanged"
    if direct_label != "correct" and rag_label == "correct":
        return "rag_fixed"
    if direct_label == "correct" and rag_label != "correct":
        return "rag_regressed"
    if direct_label == "incorrect" and rag_label == "partially_correct":
        return "rag_improved_partial"
    if direct_label == "partially_correct" and rag_label == "incorrect":
        return "rag_regressed_to_incorrect"
    return "changed"


def needs_manual_review(row: pd.Series) -> bool:
    if row["comparison_outcome"] != "unchanged":
        return True
    if row["direct_correctness_label"] != "correct":
        return True
    if row["rag_correctness_label"] != "correct":
        return True
    return False


def main() -> None:
    args = parse_args()

    direct_df = load_required_csv(args.direct_metrics)
    rag_df = load_required_csv(args.rag_metrics)
    bm25_df = load_required_csv(args.bm25_results)

    direct_columns = [
        "question_id",
        "question",
        "gold_primary",
        "predicted_answer",
        "metric_normalized_exact_match",
        "metric_token_f1",
        "correctness_label",
        "risk_label",
    ]
    rag_columns = [
        "question_id",
        "predicted_answer",
        "metric_normalized_exact_match",
        "metric_token_f1",
        "correctness_label",
        "risk_label",
        "gold_supported_in_top_k",
        "supported_gold_answer",
        "supported_rank",
    ]
    bm25_columns = [
        "question_id",
        "evidence_chunk_count",
        "gold_supported_in_top_k",
        "supported_gold_answer",
        "supported_rank",
        "retrieval_error",
    ]

    comparison_df = (
        direct_df[direct_columns]
        .rename(
            columns={
                "predicted_answer": "direct_answer",
                "metric_normalized_exact_match": "direct_normalized_exact_match",
                "metric_token_f1": "direct_token_f1",
                "correctness_label": "direct_correctness_label",
                "risk_label": "direct_risk_label",
            }
        )
        .merge(
            rag_df[rag_columns].rename(
                columns={
                    "predicted_answer": "rag_answer",
                    "metric_normalized_exact_match": "rag_normalized_exact_match",
                    "metric_token_f1": "rag_token_f1",
                    "correctness_label": "rag_correctness_label",
                    "risk_label": "rag_risk_label",
                    "gold_supported_in_top_k": "rag_gold_supported_in_top_k",
                    "supported_gold_answer": "rag_supported_gold_answer",
                    "supported_rank": "rag_supported_rank",
                }
            ),
            on="question_id",
            how="inner",
        )
        .merge(
            bm25_df[bm25_columns].rename(
                columns={
                    "gold_supported_in_top_k": "bm25_gold_supported_in_top_k",
                    "supported_gold_answer": "bm25_supported_gold_answer",
                    "supported_rank": "bm25_supported_rank",
                }
            ),
            on="question_id",
            how="left",
        )
    )

    comparison_df["token_f1_delta_rag_minus_direct"] = (
        comparison_df["rag_token_f1"] - comparison_df["direct_token_f1"]
    ).round(6)
    comparison_df["normalized_em_delta_rag_minus_direct"] = (
        comparison_df["rag_normalized_exact_match"] - comparison_df["direct_normalized_exact_match"]
    )
    comparison_df["comparison_outcome"] = comparison_df.apply(
        lambda row: outcome_bucket(row["direct_correctness_label"], row["rag_correctness_label"]),
        axis=1,
    )
    comparison_df["needs_manual_review"] = comparison_df.apply(needs_manual_review, axis=1)

    output_path = resolve_path(args.output) if args.output else resolve_path(args.tables_dir) / "pilot10_direct_vs_rag_comparison.csv"
    summary_path = (
        resolve_path(args.summary_output)
        if args.summary_output
        else resolve_path(args.tables_dir) / "pilot10_direct_vs_rag_comparison.summary.json"
    )
    review_path = (
        resolve_path(args.review_output)
        if args.review_output
        else resolve_path(args.reviews_dir) / "pilot10_manual_review_candidates.csv"
    )

    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)
    ensure_dir(review_path.parent)

    comparison_df.to_csv(output_path, index=False)

    review_columns = [
        "question_id",
        "question",
        "gold_primary",
        "direct_answer",
        "rag_answer",
        "direct_correctness_label",
        "rag_correctness_label",
        "comparison_outcome",
        "bm25_gold_supported_in_top_k",
        "bm25_supported_gold_answer",
        "bm25_supported_rank",
    ]
    review_df = comparison_df[comparison_df["needs_manual_review"]].copy()
    review_df["manual_label"] = ""
    review_df["error_type"] = ""
    review_df["review_notes"] = ""
    review_df[review_columns + ["manual_label", "error_type", "review_notes"]].to_csv(review_path, index=False)

    summary = {
        "row_count": int(len(comparison_df)),
        "comparison_file": str(output_path),
        "manual_review_file": str(review_path),
        "direct_normalized_exact_match": round(float(comparison_df["direct_normalized_exact_match"].mean()), 6),
        "rag_normalized_exact_match": round(float(comparison_df["rag_normalized_exact_match"].mean()), 6),
        "direct_mean_token_f1": round(float(comparison_df["direct_token_f1"].mean()), 6),
        "rag_mean_token_f1": round(float(comparison_df["rag_token_f1"].mean()), 6),
        "mean_token_f1_delta_rag_minus_direct": round(
            float(comparison_df["token_f1_delta_rag_minus_direct"].mean()),
            6,
        ),
        "outcome_counts": comparison_df["comparison_outcome"].value_counts().to_dict(),
        "manual_review_count": int(comparison_df["needs_manual_review"].sum()),
        "bm25_support_rate": round(float(comparison_df["bm25_gold_supported_in_top_k"].mean()), 6),
    }
    write_json(summary, summary_path)

    print(f"Wrote comparison report to {output_path}")
    print(f"Wrote comparison summary to {summary_path}")
    print(f"Wrote manual review candidates to {review_path}")


if __name__ == "__main__":
    main()
