from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_yaml_config
from src.models.direct_llm import DirectLLMConfig, create_direct_answerer
from src.utils import ensure_dir, slugify, utc_timestamp_slug, write_json


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    prompt_config = load_yaml_config("prompt_config.yaml")

    direct_defaults = experiment_config.get("direct", {})
    output_defaults = experiment_config.get("outputs", {})

    parser = argparse.ArgumentParser(description="Run the direct factual QA baseline.")
    parser.add_argument(
        "--benchmark",
        default=direct_defaults.get("benchmark_path", "data/benchmark/triviaqa_pilot_v1.csv"),
        help="Path to the benchmark CSV, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--provider",
        default=direct_defaults.get("provider", "openai"),
        choices=["openai", "reference"],
        help="Model provider. Use 'reference' only for offline pipeline verification.",
    )
    parser.add_argument(
        "--model",
        default=direct_defaults.get("model_name", "gpt-5.4-mini"),
        help="Model name for the provider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=direct_defaults.get("max_questions"),
        help="Optional limit on the number of benchmark questions to run.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(direct_defaults.get("temperature", 0.0)),
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(direct_defaults.get("max_output_tokens", 48)),
        help="Maximum output tokens for each answer.",
    )
    parser.add_argument(
        "--retry-max-output-tokens",
        type=int,
        default=int(direct_defaults.get("retry_max_output_tokens", 1024)),
        help="Maximum output tokens for one retry after an incomplete or blank answer.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=direct_defaults.get("reasoning_effort", "low"),
        help="Reasoning effort passed to supported models.",
    )
    parser.add_argument(
        "--system-prompt",
        default=prompt_config.get("direct", {}).get("system_prompt", ""),
        help="Override the configured system prompt.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV path. If omitted, a timestamped file is created in results/runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file instead of resuming from it.",
    )
    parser.add_argument(
        "--runs-dir",
        default=output_defaults.get("runs_dir", "results/runs"),
        help="Directory for generated run artifacts.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def build_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return resolve_path(args.output)

    timestamp = utc_timestamp_slug()
    filename = (
        f"direct__{slugify(args.provider)}__{slugify(args.model)}__{timestamp}.csv"
    )
    return resolve_path(args.runs_dir) / filename


def load_existing_predictions(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    return pd.read_csv(output_path)


def main() -> None:
    args = parse_args()

    benchmark_path = resolve_path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    output_path = build_output_path(args)
    ensure_dir(output_path.parent)

    benchmark_df = pd.read_csv(benchmark_path)
    if args.limit is not None:
        benchmark_df = benchmark_df.head(args.limit).copy()

    existing_df = pd.DataFrame()
    completed_question_ids: set[str] = set()
    if output_path.exists() and not args.overwrite:
        existing_df = load_existing_predictions(output_path)
        if "question_id" in existing_df.columns:
            completed_question_ids = {
                str(question_id)
                for question_id in existing_df["question_id"].dropna().astype(str).tolist()
            }

    pending_df = benchmark_df[
        ~benchmark_df["question_id"].astype(str).isin(completed_question_ids)
    ].copy()

    direct_config = DirectLLMConfig(
        model_name=args.model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        retry_max_output_tokens=args.retry_max_output_tokens,
    )
    answerer = create_direct_answerer(args.provider, direct_config)

    result_rows: list[dict[str, object]] = []
    for row in tqdm(
        pending_df.to_dict(orient="records"),
        total=len(pending_df),
        desc="Generating direct answers",
    ):
        question_text = str(row.get("question") or "").strip()
        result_row = {
            "question_id": row.get("question_id"),
            "question_source": row.get("question_source", ""),
            "question": question_text,
            "gold_primary": row.get("gold_primary"),
            "gold_aliases_json": row.get("gold_aliases_json"),
            "answer_type": row.get("answer_type", ""),
            "gold_alias_count": row.get("gold_alias_count"),
            "search_result_count": row.get("search_result_count"),
            "entity_page_count": row.get("entity_page_count"),
            "provider": args.provider,
            "model_name": args.model,
            "system_prompt": args.system_prompt,
            "predicted_answer": "",
            "response_id": "",
            "response_status": "",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "retry_count": 0,
            "final_max_output_tokens": args.max_output_tokens,
            "generation_error": "",
            "generated_at_utc": utc_timestamp_slug(),
        }

        try:
            generation = answerer.generate_answer(question_text, row)
            result_row.update(
                {
                    "predicted_answer": generation.answer_text,
                    "response_id": generation.response_id,
                    "response_status": generation.response_status,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                    "total_tokens": generation.total_tokens,
                    "retry_count": generation.retry_count,
                    "final_max_output_tokens": generation.final_max_output_tokens,
                    "generation_error": generation.error_message,
                }
            )
        except Exception as exc:  # noqa: BLE001
            result_row["generation_error"] = str(exc)

        result_rows.append(result_row)

        combined_df = pd.concat(
            [existing_df, pd.DataFrame(result_rows)],
            ignore_index=True,
        )
        combined_df.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "run_type": "direct",
        "provider": args.provider,
        "model_name": args.model,
        "system_prompt": args.system_prompt,
        "benchmark_path": str(benchmark_path),
        "output_path": str(output_path),
        "requested_rows": int(len(benchmark_df)),
        "completed_rows": int(len(existing_df) + len(result_rows)),
        "resumed_rows": int(len(existing_df)),
        "new_rows": int(len(result_rows)),
        "failed_rows": int(
            sum(
                1
                for row in pd.concat([existing_df, pd.DataFrame(result_rows)], ignore_index=True).to_dict(
                    orient="records"
                )
                if str(row.get("generation_error") or "").strip()
            )
        ),
    }
    write_json(summary, summary_path)

    print(f"Wrote direct baseline predictions to {output_path}")
    print(f"Wrote run summary to {summary_path}")


if __name__ == "__main__":
    main()
