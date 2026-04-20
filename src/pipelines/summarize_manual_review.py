from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_yaml_config
from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    output_defaults = experiment_config.get("outputs", {})

    parser = argparse.ArgumentParser(description="Summarize manual review labels.")
    parser.add_argument(
        "--review-file",
        default="results/manual_reviews/pilot10_manual_review_candidates.csv",
        help="Manual review CSV path, relative to project root unless absolute.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional summary JSON path.",
    )
    parser.add_argument(
        "--tables-dir",
        default=output_defaults.get("tables_dir", "results/tables"),
        help="Directory for summary output.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()

    review_path = resolve_path(args.review_file)
    if not review_path.exists():
        raise FileNotFoundError(f"Manual review file not found: {review_path}")

    review_df = pd.read_csv(review_path, keep_default_na=False)
    labeled_df = review_df[review_df["manual_label"].str.strip() != ""].copy()

    output_path = (
        resolve_path(args.output)
        if args.output
        else resolve_path(args.tables_dir) / f"{review_path.stem}.summary.json"
    )
    ensure_dir(output_path.parent)

    summary = {
        "review_file": str(review_path),
        "row_count": int(len(review_df)),
        "labeled_count": int(len(labeled_df)),
        "unlabeled_count": int(len(review_df) - len(labeled_df)),
        "manual_label_counts": labeled_df["manual_label"].value_counts().to_dict(),
        "error_type_counts": labeled_df["error_type"].value_counts().to_dict(),
        "comparison_outcome_counts": review_df["comparison_outcome"].value_counts().to_dict(),
    }
    write_json(summary, output_path)

    print(f"Wrote manual review summary to {output_path}")


if __name__ == "__main__":
    main()
