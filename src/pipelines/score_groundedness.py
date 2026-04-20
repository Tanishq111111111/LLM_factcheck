from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_yaml_config
from src.data.normalize_answers import normalize_text
from src.utils import ensure_dir, write_json


INSUFFICIENT_EVIDENCE_PATTERN = re.compile(r"[\W_]+")


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    output_defaults = experiment_config.get("outputs", {})

    parser = argparse.ArgumentParser(description="Score lexical groundedness proxies for a RAG run.")
    parser.add_argument(
        "--rag-run",
        default="results/runs/rag__openai__gpt-5-4-mini__bm25_top5__pilot100.csv",
        help="RAG prediction CSV path.",
    )
    parser.add_argument(
        "--rag-metrics",
        default="results/tables/rag-openai-gpt-5-4-mini-bm25-top5-pilot100__metrics.csv",
        help="RAG row-level metrics CSV path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional groundedness row-level CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional groundedness summary JSON path.",
    )
    parser.add_argument(
        "--tables-dir",
        default=output_defaults.get("tables_dir", "results/tables"),
        help="Directory for groundedness artifacts.",
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
    return pd.read_csv(path, keep_default_na=False)


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)]


def is_insufficient_evidence(answer: str) -> bool:
    compact_answer = INSUFFICIENT_EVIDENCE_PATTERN.sub("", str(answer)).lower()
    return compact_answer == "insufficientevidence"


def normalized_phrase_in_text(phrase: str, text: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    normalized_text = normalize_text(text)
    if not normalized_phrase or not normalized_text:
        return False
    return normalized_phrase in normalized_text


def find_supporting_passage(answer: str, passages: list[str]) -> tuple[bool, int | None]:
    for index, passage in enumerate(passages, start=1):
        if normalized_phrase_in_text(answer, passage):
            return True, index
    return False, None


def score_row(row: pd.Series) -> dict[str, object]:
    predicted_answer = str(row.get("predicted_answer") or "").strip()
    correctness_label = str(row.get("correctness_label") or "").strip()
    evidence_passages = parse_json_list(row.get("evidence_passages_json"))
    evidence_text = " ".join(evidence_passages)

    prediction_is_empty = predicted_answer == ""
    prediction_is_insufficient = is_insufficient_evidence(predicted_answer)
    answer_supported, support_rank = find_supporting_passage(predicted_answer, evidence_passages)
    gold_supported = str(row.get("gold_supported_in_top_k")).lower() == "true"

    supported_or_abstained = answer_supported or prediction_is_insufficient
    unsupported_answer = not prediction_is_empty and not prediction_is_insufficient and not answer_supported
    refused_despite_gold_support = prediction_is_insufficient and gold_supported

    if prediction_is_empty:
        groundedness_bucket = "empty_prediction"
    elif refused_despite_gold_support:
        groundedness_bucket = "abstained_despite_gold_support"
    elif prediction_is_insufficient:
        groundedness_bucket = "abstained_without_gold_support"
    elif unsupported_answer:
        groundedness_bucket = "unsupported_answer_proxy"
    elif correctness_label == "correct":
        groundedness_bucket = "supported_correct"
    elif correctness_label == "partially_correct":
        groundedness_bucket = "supported_partial"
    else:
        groundedness_bucket = "supported_but_incorrect"

    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "gold_primary": row.get("gold_primary"),
        "predicted_answer": predicted_answer,
        "correctness_label": correctness_label,
        "metric_normalized_exact_match": row.get("metric_normalized_exact_match", ""),
        "metric_token_f1": row.get("metric_token_f1", ""),
        "gold_supported_in_top_k": gold_supported,
        "supported_gold_answer": row.get("supported_gold_answer", ""),
        "supported_rank": row.get("supported_rank", ""),
        "prediction_is_empty": prediction_is_empty,
        "prediction_is_insufficient_evidence": prediction_is_insufficient,
        "answer_supported_by_retrieved_evidence": answer_supported,
        "answer_support_passage_rank": support_rank if support_rank is not None else "",
        "supported_or_abstained": supported_or_abstained,
        "unsupported_answer_proxy": unsupported_answer,
        "refused_despite_gold_support": refused_despite_gold_support,
        "groundedness_bucket": groundedness_bucket,
        "evidence_passage_count": len(evidence_passages),
        "evidence_char_count": len(evidence_text),
    }


def main() -> None:
    args = parse_args()

    rag_df = load_required_csv(args.rag_run)
    metrics_df = load_required_csv(args.rag_metrics)

    metric_columns = [
        "question_id",
        "correctness_label",
        "metric_normalized_exact_match",
        "metric_token_f1",
    ]
    missing_metric_columns = [column for column in metric_columns if column not in metrics_df.columns]
    if missing_metric_columns:
        raise ValueError(f"RAG metrics file is missing columns: {missing_metric_columns}")

    merged_df = rag_df.merge(metrics_df[metric_columns], on="question_id", how="left")
    groundedness_df = pd.DataFrame([score_row(row) for _, row in merged_df.iterrows()])

    tables_dir = resolve_path(args.tables_dir)
    output_path = (
        resolve_path(args.output)
        if args.output
        else tables_dir / "pilot100_rag_groundedness_analysis.csv"
    )
    summary_path = (
        resolve_path(args.summary_output)
        if args.summary_output
        else tables_dir / "pilot100_rag_groundedness_analysis.summary.json"
    )
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)

    groundedness_df.to_csv(output_path, index=False)

    answered_df = groundedness_df[
        (~groundedness_df["prediction_is_empty"]) & (~groundedness_df["prediction_is_insufficient_evidence"])
    ]
    summary = {
        "rag_run_file": str(resolve_path(args.rag_run)),
        "rag_metrics_file": str(resolve_path(args.rag_metrics)),
        "groundedness_file": str(output_path),
        "row_count": int(len(groundedness_df)),
        "method": "lexical_answer_overlap_with_retrieved_passages",
        "limitations": [
            "This is a conservative proxy, not a human factuality judgment.",
            "It can miss paraphrased support and can count misleading lexical overlap as support.",
        ],
        "aggregate_metrics": {
            "gold_support_at_k": round(float(groundedness_df["gold_supported_in_top_k"].mean()), 6),
            "answer_supported_rate_all_rows": round(
                float(groundedness_df["answer_supported_by_retrieved_evidence"].mean()),
                6,
            ),
            "answer_supported_rate_answered_rows": round(
                float(answered_df["answer_supported_by_retrieved_evidence"].mean()) if len(answered_df) else 0.0,
                6,
            ),
            "unsupported_answer_proxy_rate_all_rows": round(
                float(groundedness_df["unsupported_answer_proxy"].mean()),
                6,
            ),
            "refused_despite_gold_support_rate": round(
                float(groundedness_df["refused_despite_gold_support"].mean()),
                6,
            ),
            "supported_but_incorrect_rate": round(
                float((groundedness_df["groundedness_bucket"] == "supported_but_incorrect").mean()),
                6,
            ),
            "supported_partial_rate": round(
                float((groundedness_df["groundedness_bucket"] == "supported_partial").mean()),
                6,
            ),
            "insufficient_evidence_rate": round(
                float(groundedness_df["prediction_is_insufficient_evidence"].mean()),
                6,
            ),
        },
        "unsupported_answer_count": int(groundedness_df["unsupported_answer_proxy"].sum()),
        "refused_despite_gold_support_count": int(groundedness_df["refused_despite_gold_support"].sum()),
        "groundedness_bucket_counts": groundedness_df["groundedness_bucket"].value_counts().to_dict(),
        "correctness_counts": groundedness_df["correctness_label"].value_counts().to_dict(),
    }
    write_json(summary, summary_path)

    print(f"Wrote groundedness analysis to {output_path}")
    print(f"Wrote groundedness summary to {summary_path}")
    print(
        "Answer support rate: "
        f"{summary['aggregate_metrics']['answer_supported_rate_all_rows']:.3f} | "
        "Unsupported proxy rate: "
        f"{summary['aggregate_metrics']['unsupported_answer_proxy_rate_all_rows']:.3f}"
    )


if __name__ == "__main__":
    main()
