from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_yaml_config
from src.data.normalize_answers import normalize_text
from src.utils import ensure_dir


INSUFFICIENT_EVIDENCE = "insufficient_evidence"
ABBREVIATION_PATTERN = re.compile(r"^[A-Z0-9]{1,5}$")
YEAR_RANGE_PATTERN = re.compile(r"\b\d{3,4}\s*[-–]\s*\d{2,4}\b")


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    output_defaults = experiment_config.get("outputs", {})

    parser = argparse.ArgumentParser(description="Create a pre-labeled manual review draft.")
    parser.add_argument(
        "--input",
        default="results/manual_reviews/pilot100_manual_review_candidates.csv",
        help="Manual review candidate CSV path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional labeled review output CSV path.",
    )
    parser.add_argument(
        "--reviews-dir",
        default=output_defaults.get("reviews_dir", "results/manual_reviews"),
        help="Directory for review artifacts.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def is_insufficient(answer: str) -> bool:
    return normalize_text(answer) == INSUFFICIENT_EVIDENCE


def is_abbreviation_answer(gold_answer: str) -> bool:
    return bool(ABBREVIATION_PATTERN.match(gold_answer.strip()))


def is_overcomplete(gold_answer: str, answer: str) -> bool:
    normalized_gold = normalize_text(gold_answer)
    normalized_answer = normalize_text(answer)
    if not normalized_gold or not normalized_answer:
        return False
    if normalized_answer == normalized_gold:
        return False
    return normalized_gold in normalized_answer and len(normalized_answer.split()) > len(normalized_gold.split())


def is_plural_or_morphology_variant(gold_answer: str, answer: str) -> bool:
    normalized_gold = normalize_text(gold_answer)
    normalized_answer = normalize_text(answer)
    if not normalized_gold or not normalized_answer:
        return False
    if normalized_gold == normalized_answer:
        return True
    return normalized_gold.rstrip("s") == normalized_answer.rstrip("s")


def has_date_or_current_language(question: str, gold_answer: str) -> bool:
    question_text = question.lower()
    if "current" in question_text or "currently" in question_text:
        return True
    return bool(YEAR_RANGE_PATTERN.search(gold_answer))


def assign_prelabels(row: pd.Series) -> tuple[str, str, str]:
    question = safe_text(row.get("question"))
    gold = safe_text(row.get("gold_primary"))
    direct = safe_text(row.get("direct_answer"))
    rag = safe_text(row.get("rag_answer"))
    direct_label = safe_text(row.get("direct_correctness_label"))
    rag_label = safe_text(row.get("rag_correctness_label"))
    outcome = safe_text(row.get("comparison_outcome"))
    bm25_supported = str(row.get("bm25_gold_supported_in_top_k")).lower() == "true"

    if outcome == "rag_fixed":
        if has_date_or_current_language(question, gold):
            return (
                "direct_wrong_rag_correct",
                "outdated_or_context_sensitive_fact",
                "RAG corrected a direct answer that appears sensitive to time or benchmark context.",
            )
        if direct_label == "partially_correct":
            return (
                "direct_partial_rag_correct",
                "wrong_granularity",
                "RAG returned the expected shorter gold answer while direct was over-specific or differently phrased.",
            )
        return (
            "direct_wrong_rag_correct",
            "incorrect_entity",
            "RAG corrected a direct-answer mistake.",
        )

    if outcome in {"rag_regressed", "rag_regressed_to_incorrect"}:
        if is_insufficient(rag):
            return (
                "direct_correct_rag_wrong",
                "insufficient_evidence_refusal" if bm25_supported else "retrieval_failure",
                "RAG refused to answer despite direct being better; check whether retrieved evidence was salient enough.",
            )
        if is_overcomplete(gold, rag):
            return (
                "direct_correct_rag_partial",
                "overcomplete_answer",
                "RAG included extra information beyond the expected answer.",
            )
        return (
            "direct_correct_rag_wrong",
            "incorrect_entity",
            "RAG changed a better direct answer into a worse answer.",
        )

    if direct_label == "incorrect" and rag_label == "incorrect":
        if has_date_or_current_language(question, gold):
            return (
                "both_wrong",
                "outdated_or_context_sensitive_fact",
                "Both systems conflict with the benchmark answer on a time-sensitive or context-sensitive question.",
            )
        if bm25_supported:
            return (
                "both_wrong",
                "evidence_use_failure",
                "BM25 found supporting evidence, but neither system returned the benchmark answer.",
            )
        return (
            "both_wrong",
            "retrieval_failure",
            "Neither system returned the benchmark answer and BM25 did not find answer support.",
        )

    if direct_label == "partially_correct" and rag_label == "partially_correct":
        if has_date_or_current_language(question, gold):
            return (
                "ambiguous_partial",
                "historical_geography_ambiguity",
                "Both answers are partially aligned with an ambiguous or historically sensitive gold answer.",
            )
        if is_plural_or_morphology_variant(gold, direct) or is_plural_or_morphology_variant(gold, rag):
            return (
                "metric_artifact",
                "metric_morphology",
                "At least one answer appears to be a morphology or plurality variant that automatic scoring treats as partial.",
            )
        return (
            "ambiguous_partial",
            "wrong_granularity",
            "Both systems are close but differ from the expected answer granularity.",
        )

    if direct_label == "correct" and rag_label == "correct":
        return (
            "both_correct",
            "none",
            "Both systems match the benchmark answer.",
        )

    if direct_label == "partially_correct" and rag_label == "incorrect":
        return (
            "direct_partial_rag_wrong",
            "incorrect_entity",
            "RAG moved from a partially correct direct answer to an incorrect answer.",
        )

    if direct_label == "incorrect" and rag_label == "partially_correct":
        return (
            "direct_wrong_rag_partial",
            "wrong_granularity",
            "RAG improved the direct failure but did not fully match the gold answer.",
        )

    if is_abbreviation_answer(gold) and rag_label != "correct":
        return (
            "needs_human_check",
            "abbreviation_confusion",
            "Gold answer is an abbreviation; verify whether retrieved evidence supports the abbreviation or a misleading expansion.",
        )

    return (
        "needs_human_check",
        "uncategorized",
        "Review manually; this row did not match a simple pre-labeling rule.",
    )


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Manual review file not found: {input_path}")

    output_path = (
        resolve_path(args.output)
        if args.output
        else resolve_path(args.reviews_dir) / f"{input_path.stem}_prelabeled.csv"
    )
    ensure_dir(output_path.parent)

    review_df = pd.read_csv(input_path, keep_default_na=False)
    labels = review_df.apply(assign_prelabels, axis=1)
    review_df["manual_label"] = [label[0] for label in labels]
    review_df["error_type"] = [label[1] for label in labels]
    review_df["review_notes"] = [label[2] for label in labels]
    review_df["review_status"] = "prelabeled_needs_human_validation"

    review_df.to_csv(output_path, index=False)
    print(f"Wrote pre-labeled review draft to {output_path}")


if __name__ == "__main__":
    main()
