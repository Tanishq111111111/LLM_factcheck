from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final-report-ready results summaries.")
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
        "--output-md",
        default="results/tables/pilot100_results_summary.md",
        help="Markdown results summary output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tables/pilot100_results_summary.csv",
        help="Compact CSV results table output path.",
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


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def percentage_points(value: float) -> str:
    return f"{value * 100:+.1f} percentage points"


def format_counts(counts: dict[str, int]) -> str:
    ordered_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{label}: {count}" for label, count in ordered_items)


def main() -> None:
    args = parse_args()

    comparison = load_json(args.comparison_summary)
    bm25 = load_json(args.bm25_summary)
    manual_review = load_json(args.manual_review_summary)
    groundedness = load_json(args.groundedness_summary)

    output_md = resolve_path(args.output_md)
    output_csv = resolve_path(args.output_csv)
    ensure_dir(output_md.parent)
    ensure_dir(output_csv.parent)

    direct_em = comparison["direct_normalized_exact_match"]
    rag_em = comparison["rag_normalized_exact_match"]
    direct_f1 = comparison["direct_mean_token_f1"]
    rag_f1 = comparison["rag_mean_token_f1"]
    em_delta = rag_em - direct_em
    f1_delta = comparison["mean_token_f1_delta_rag_minus_direct"]

    outcome_counts = comparison["outcome_counts"]
    groundedness_metrics = groundedness["aggregate_metrics"]
    groundedness_buckets = groundedness["groundedness_bucket_counts"]
    manual_error_counts = manual_review["error_type_counts"]
    most_common_error = max(manual_error_counts, key=manual_error_counts.get)

    csv_lines = [
        "system,normalized_exact_match,mean_token_f1,support_at_5,answer_support_rate,unsupported_answer_proxy_rate,notes",
        f"Direct LLM,{direct_em:.6f},{direct_f1:.6f},,,,No retrieved evidence",
        f"BM25 Retrieval,,,{bm25['gold_support_rate_top_k']:.6f},,,Gold answer appears in top-5 retrieved passages",
        (
            f"BM25 + RAG,{rag_em:.6f},{rag_f1:.6f},,"
            f"{groundedness_metrics['answer_supported_rate_all_rows']:.6f},"
            f"{groundedness_metrics['unsupported_answer_proxy_rate_all_rows']:.6f},"
            "Uses BM25 top-5 evidence"
        ),
    ]
    output_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    md = f"""# Pilot 100 Results Summary

## Main Result

Retrieval grounding produced a modest reliability gain on the 100-question TriviaQA pilot benchmark.

| System | Normalized EM | Mean Token F1 | Support@5 |
|---|---:|---:|---:|
| Direct LLM | {direct_em:.3f} | {direct_f1:.3f} | |
| BM25 Retrieval | | | {bm25['gold_support_rate_top_k']:.3f} |
| BM25 + RAG | {rag_em:.3f} | {rag_f1:.3f} | |

## Interpretation

- RAG improved normalized exact match by {em_delta:+.3f} ({percentage_points(em_delta)}).
- RAG improved mean token F1 by {f1_delta:+.3f}.
- BM25 retrieved answer-supporting evidence in {percent(bm25['gold_support_rate_top_k'])} of benchmark rows.
- RAG fixed {outcome_counts.get('rag_fixed', 0)} direct failures and improved {outcome_counts.get('rag_improved_partial', 0)} additional cases to partial correctness.
- RAG regressed {outcome_counts.get('rag_regressed', 0) + outcome_counts.get('rag_regressed_to_incorrect', 0)} cases, showing that evidence retrieval alone does not guarantee better answer generation.
- {comparison['manual_review_count']} rows were flagged for manual review before final metric or prompt changes.

## Groundedness Proxy

- Gold answer support@5 was {percent(bm25['gold_support_rate_top_k'])}, so retrieval usually found relevant evidence.
- The RAG answer string appeared in retrieved evidence for {percent(groundedness_metrics['answer_supported_rate_all_rows'])} of all rows.
- Among non-empty, non-abstention RAG answers, lexical answer support was {percent(groundedness_metrics['answer_supported_rate_answered_rows'])}.
- Unsupported-answer proxy rate was {percent(groundedness_metrics['unsupported_answer_proxy_rate_all_rows'])}.
- RAG abstained despite top-k gold support in {groundedness['refused_despite_gold_support_count']} rows.
- Groundedness buckets: {format_counts(groundedness_buckets)}.

## Manual Review

- Final reviewed rows: {manual_review['labeled_count']} of {manual_review['row_count']}.
- Most common reviewed error type: `{most_common_error}` ({manual_error_counts[most_common_error]} rows).
- Manual labels: {format_counts(manual_review['manual_label_counts'])}.

## Current Conclusion

The project now has evidence that retrieval grounding can improve factual QA reliability, but the improvement is not automatic. The strongest failure pattern is not simple lack of evidence: BM25 often retrieves useful evidence, while RAG can still select the wrong entity, over-answer, or refuse despite support. The next step is to turn these findings into final figures and a concise report narrative.
"""
    output_md.write_text(md, encoding="utf-8")

    print(f"Wrote markdown summary to {output_md}")
    print(f"Wrote compact results table to {output_csv}")


if __name__ == "__main__":
    main()
