from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_yaml_config
from src.utils import ensure_dir


REQUIRED_COLUMNS = {
    "question_id",
    "manual_label",
    "error_type",
    "review_notes",
    "review_status",
}


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    output_defaults = experiment_config.get("outputs", {})

    parser = argparse.ArgumentParser(description="Finalize a human-validated manual review CSV.")
    parser.add_argument(
        "--input",
        default="results/manual_reviews/pilot100_manual_review_candidates_prelabeled.csv",
        help="Validated pre-labeled review CSV path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Final reviewed CSV path.",
    )
    parser.add_argument(
        "--reviews-dir",
        default=output_defaults.get("reviews_dir", "results/manual_reviews"),
        help="Directory for final manual review artifacts.",
    )
    parser.add_argument(
        "--review-status",
        default="human_verified",
        help="Review status value to apply to all rows.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def validate_review_columns(review_df: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS.difference(review_df.columns))
    if missing_columns:
        raise ValueError(f"Manual review file is missing columns: {missing_columns}")

    unlabeled_rows = review_df[
        (review_df["manual_label"].fillna("").str.strip() == "")
        | (review_df["error_type"].fillna("").str.strip() == "")
    ]
    if not unlabeled_rows.empty:
        unlabeled_ids = unlabeled_rows["question_id"].astype(str).tolist()
        raise ValueError(f"Manual review still has unlabeled rows: {unlabeled_ids}")


def main() -> None:
    args = parse_args()

    input_path = resolve_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Manual review file not found: {input_path}")

    output_path = (
        resolve_path(args.output)
        if args.output
        else resolve_path(args.reviews_dir) / "pilot100_manual_review_final.csv"
    )
    ensure_dir(output_path.parent)

    review_df = pd.read_csv(input_path, keep_default_na=False)
    validate_review_columns(review_df)

    finalized_df = review_df.copy()
    finalized_df["review_status"] = args.review_status
    finalized_df.to_csv(output_path, index=False)

    print(f"Wrote finalized manual review to {output_path}")
    print(f"Finalized rows: {len(finalized_df)}")


if __name__ == "__main__":
    main()
