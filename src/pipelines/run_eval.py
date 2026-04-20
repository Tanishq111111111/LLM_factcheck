from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_yaml_config
from src.evaluation.labels import CorrectnessLabel, RiskLabel
from src.evaluation.metrics import best_token_f1, exact_match, normalized_exact_match
from src.utils import ensure_dir, slugify, utc_timestamp_slug, write_json


SUPPORTED_METRICS = {
    "exact_match",
    "normalized_exact_match",
    "token_f1",
}


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    output_defaults = experiment_config.get("outputs", {})
    evaluation_defaults = experiment_config.get("evaluation", {})

    parser = argparse.ArgumentParser(description="Evaluate prediction outputs against the benchmark.")
    parser.add_argument(
        "--predictions",
        default="",
        help="Prediction CSV path. If omitted, the latest direct run in results/runs is used.",
    )
    parser.add_argument(
        "--benchmark",
        default=evaluation_defaults.get("benchmark_path", "data/benchmark/triviaqa_pilot_v1.csv"),
        help="Benchmark CSV path, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--tables-dir",
        default=output_defaults.get("tables_dir", "results/tables"),
        help="Directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--partial-f1-threshold",
        type=float,
        default=float(evaluation_defaults.get("partial_f1_threshold", 0.5)),
        help="Threshold above which a non-exact answer is labeled partially_correct.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix override for evaluation outputs.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def find_latest_prediction_file(runs_dir: Path) -> Path:
    candidates = sorted(
        runs_dir.glob("direct__*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No direct run CSV files found in {runs_dir}")
    return candidates[0]


def parse_json_list(value) -> list[str]:
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


def assign_correctness_label(normalized_em: bool, token_f1_value: float, prediction: str, threshold: float) -> str:
    if normalized_em:
        return CorrectnessLabel.CORRECT.value
    if prediction.strip() and token_f1_value >= threshold:
        return CorrectnessLabel.PARTIALLY_CORRECT.value
    return CorrectnessLabel.INCORRECT.value


def assign_risk_label(correctness_label: str) -> str:
    if correctness_label == CorrectnessLabel.CORRECT.value:
        return RiskLabel.LOW.value
    if correctness_label == CorrectnessLabel.PARTIALLY_CORRECT.value:
        return RiskLabel.MEDIUM.value
    return RiskLabel.HIGH.value


def build_output_prefix(predictions_path: Path, override: str) -> str:
    if override:
        return slugify(override)
    return slugify(predictions_path.stem)


def main() -> None:
    args = parse_args()
    experiment_config = load_yaml_config("experiment_config.yaml")
    requested_metrics = experiment_config.get("metrics", [])

    benchmark_path = resolve_path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    runs_dir = resolve_path(experiment_config.get("outputs", {}).get("runs_dir", "results/runs"))
    predictions_path = resolve_path(args.predictions) if args.predictions else find_latest_prediction_file(runs_dir)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    tables_dir = resolve_path(args.tables_dir)
    ensure_dir(tables_dir)

    benchmark_df = pd.read_csv(benchmark_path)
    predictions_df = pd.read_csv(predictions_path)

    benchmark_columns = [
        "question_id",
        "question",
        "gold_primary",
        "gold_aliases_json",
        "question_source",
    ]
    missing_prediction_columns = [column for column in benchmark_columns if column not in predictions_df.columns]
    if missing_prediction_columns:
        predictions_df = predictions_df.merge(
            benchmark_df[benchmark_columns],
            on="question_id",
            how="left",
            suffixes=("", "_benchmark"),
        )

    evaluated_rows: list[dict[str, object]] = []
    for row in predictions_df.to_dict(orient="records"):
        gold_answers = parse_json_list(row.get("gold_aliases_json"))
        prediction_text = str(row.get("predicted_answer") or "").strip()

        exact_match_value = exact_match(prediction_text, gold_answers)
        normalized_em_value = normalized_exact_match(prediction_text, gold_answers)
        token_f1_value = best_token_f1(prediction_text, gold_answers)

        if gold_answers:
            best_match = max(gold_answers, key=lambda answer: best_token_f1(prediction_text, [answer]))
        else:
            best_match = ""

        correctness_label = assign_correctness_label(
            normalized_em=normalized_em_value,
            token_f1_value=token_f1_value,
            prediction=prediction_text,
            threshold=args.partial_f1_threshold,
        )
        risk_label = assign_risk_label(correctness_label)

        evaluated_row = dict(row)
        evaluated_row.update(
            {
                "best_matching_gold_answer": best_match,
                "metric_exact_match": int(exact_match_value),
                "metric_normalized_exact_match": int(normalized_em_value),
                "metric_token_f1": round(float(token_f1_value), 6),
                "correctness_label": correctness_label,
                "risk_label": risk_label,
                "evaluation_version": "v1",
                "evaluated_at_utc": utc_timestamp_slug(),
            }
        )
        evaluated_rows.append(evaluated_row)

    evaluated_df = pd.DataFrame(evaluated_rows)
    if "generation_error" not in evaluated_df.columns:
        evaluated_df["generation_error"] = ""
    if "predicted_answer" not in evaluated_df.columns:
        evaluated_df["predicted_answer"] = ""

    prefix = build_output_prefix(predictions_path, args.output_prefix)
    row_metrics_path = tables_dir / f"{prefix}__metrics.csv"
    summary_path = tables_dir / f"{prefix}__summary.json"

    evaluated_df.to_csv(row_metrics_path, index=False)

    computed_metrics = [metric_name for metric_name in requested_metrics if metric_name in SUPPORTED_METRICS]
    skipped_metrics = [metric_name for metric_name in requested_metrics if metric_name not in SUPPORTED_METRICS]
    summary = {
        "prediction_file": str(predictions_path),
        "benchmark_file": str(benchmark_path),
        "row_metrics_file": str(row_metrics_path),
        "row_count": int(len(evaluated_df)),
        "requested_metrics": requested_metrics,
        "computed_metrics": computed_metrics,
        "skipped_metrics": skipped_metrics,
        "partial_f1_threshold": args.partial_f1_threshold,
        "aggregate_metrics": {
            "exact_match_rate": round(float(evaluated_df["metric_exact_match"].mean()), 6),
            "normalized_exact_match_rate": round(
                float(evaluated_df["metric_normalized_exact_match"].mean()),
                6,
            ),
            "mean_token_f1": round(float(evaluated_df["metric_token_f1"].mean()), 6),
            "empty_prediction_rate": round(
                float((evaluated_df["predicted_answer"].fillna("").str.strip() == "").mean()),
                6,
            ),
            "generation_error_rate": round(
                float((evaluated_df["generation_error"].fillna("").str.strip() != "").mean()),
                6,
            ),
        },
        "label_counts": {
            "correctness": evaluated_df["correctness_label"].value_counts().to_dict(),
            "risk": evaluated_df["risk_label"].value_counts().to_dict(),
        },
    }
    write_json(summary, summary_path)

    print(f"Wrote row-level metrics to {row_metrics_path}")
    print(f"Wrote evaluation summary to {summary_path}")
    print(
        "Normalized EM: "
        f"{summary['aggregate_metrics']['normalized_exact_match_rate']:.3f} | "
        f"Mean Token F1: {summary['aggregate_metrics']['mean_token_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
