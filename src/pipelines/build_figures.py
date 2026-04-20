from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT, load_yaml_config
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    output_defaults = experiment_config.get("outputs", {})

    parser = argparse.ArgumentParser(description="Build final-report figures for the pilot results.")
    parser.add_argument(
        "--comparison-summary",
        default="results/tables/pilot100_direct_vs_rag_comparison.summary.json",
        help="Direct-vs-RAG comparison summary JSON.",
    )
    parser.add_argument(
        "--bm25-summary",
        default="results/runs/bm25__triviaqa_pilot_v1__top5.summary.json",
        help="BM25 retrieval summary JSON.",
    )
    parser.add_argument(
        "--manual-review-summary",
        default="results/tables/pilot100_manual_review_final.summary.json",
        help="Final manual review summary JSON.",
    )
    parser.add_argument(
        "--groundedness-summary",
        default="results/tables/pilot100_rag_groundedness_analysis.summary.json",
        help="RAG groundedness proxy summary JSON.",
    )
    parser.add_argument(
        "--figures-dir",
        default=output_defaults.get("figures_dir", "results/figures"),
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_json(path_text: str) -> dict:
    path = resolve_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Required summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
    horizontal: bool = False,
) -> None:
    plt.figure(figsize=(10, 5.8))
    if horizontal:
        positions = range(len(labels))
        plt.barh(positions, values, color=color)
        plt.yticks(positions, labels)
        plt.xlabel(ylabel)
        plt.gca().invert_yaxis()
        for index, value in enumerate(values):
            plt.text(value + max(values) * 0.01, index, f"{value:g}", va="center", fontsize=9)
    else:
        positions = range(len(labels))
        plt.bar(positions, values, color=color)
        plt.xticks(positions, labels, rotation=20, ha="right")
        plt.ylabel(ylabel)
        plt.ylim(0, max(1.0, max(values) * 1.15))
        for index, value in enumerate(values):
            plt.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def sorted_counts(counts: dict[str, int]) -> tuple[list[str], list[int]]:
    ordered_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [item[0] for item in ordered_items], [int(item[1]) for item in ordered_items]


def main() -> None:
    args = parse_args()

    comparison = load_json(args.comparison_summary)
    bm25 = load_json(args.bm25_summary)
    manual_review = load_json(args.manual_review_summary)
    groundedness = load_json(args.groundedness_summary)

    figures_dir = resolve_path(args.figures_dir)
    ensure_dir(figures_dir)

    metric_labels = [
        "Direct EM",
        "RAG EM",
        "Direct F1",
        "RAG F1",
        "BM25 Support@5",
    ]
    metric_values = [
        float(comparison["direct_normalized_exact_match"]),
        float(comparison["rag_normalized_exact_match"]),
        float(comparison["direct_mean_token_f1"]),
        float(comparison["rag_mean_token_f1"]),
        float(bm25["gold_support_rate_top_k"]),
    ]
    save_bar_chart(
        labels=metric_labels,
        values=metric_values,
        title="Pilot 100: Direct vs Retrieval-Grounded Reliability",
        ylabel="Rate / Score",
        output_path=figures_dir / "pilot100_metric_comparison.png",
        color="#2f6f73",
    )

    error_labels, error_values = sorted_counts(manual_review["error_type_counts"])
    save_bar_chart(
        labels=error_labels,
        values=error_values,
        title="Pilot 100 Manual Review: Error Types",
        ylabel="Rows",
        output_path=figures_dir / "pilot100_manual_review_error_types.png",
        color="#b55d39",
        horizontal=True,
    )

    bucket_labels, bucket_values = sorted_counts(groundedness["groundedness_bucket_counts"])
    save_bar_chart(
        labels=bucket_labels,
        values=bucket_values,
        title="Pilot 100 RAG Groundedness Buckets",
        ylabel="Rows",
        output_path=figures_dir / "pilot100_groundedness_buckets.png",
        color="#4d5f9f",
        horizontal=True,
    )

    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
