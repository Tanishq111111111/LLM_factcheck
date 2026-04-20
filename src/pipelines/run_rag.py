from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_yaml_config
from src.models.rag_llm import RagLLMConfig, create_rag_answerer
from src.utils import ensure_dir, slugify, utc_timestamp_slug, write_json


def parse_args() -> argparse.Namespace:
    experiment_config = load_yaml_config("experiment_config.yaml")
    prompt_config = load_yaml_config("prompt_config.yaml")
    rag_defaults = experiment_config.get("rag", {})
    output_defaults = experiment_config.get("outputs", {})

    parser = argparse.ArgumentParser(description="Run retrieval-grounded factual QA over BM25 passages.")
    parser.add_argument(
        "--retrieval",
        default=rag_defaults.get("retrieval_path", "results/runs/bm25__triviaqa_pilot_v1__top5.csv"),
        help="BM25 retrieval CSV path, relative to project root unless absolute.",
    )
    parser.add_argument(
        "--provider",
        default=rag_defaults.get("provider", "openai"),
        choices=["openai", "reference"],
        help="Model provider. Use 'reference' only for offline pipeline verification.",
    )
    parser.add_argument(
        "--model",
        default=rag_defaults.get("model_name", "gpt-5.4-mini"),
        help="Model name for the provider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=rag_defaults.get("max_questions"),
        help="Optional limit on retrieved questions.",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=int(rag_defaults.get("max_passages", 5)),
        help="Number of retrieved passages included in the prompt.",
    )
    parser.add_argument(
        "--max-passage-chars",
        type=int,
        default=int(rag_defaults.get("max_passage_chars", 1200)),
        help="Maximum characters from each passage included in the prompt.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(rag_defaults.get("max_output_tokens", 96)),
        help="Maximum output tokens for the first model attempt.",
    )
    parser.add_argument(
        "--retry-max-output-tokens",
        type=int,
        default=int(rag_defaults.get("retry_max_output_tokens", 1024)),
        help="Maximum output tokens for one retry after incomplete or blank output.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=rag_defaults.get("reasoning_effort", "low"),
        help="Reasoning effort passed to supported models.",
    )
    parser.add_argument(
        "--system-prompt",
        default=prompt_config.get("rag", {}).get("system_prompt", ""),
        help="Override the configured RAG system prompt.",
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
        help="Directory for generated RAG artifacts.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def build_output_path(args: argparse.Namespace, retrieval_path: Path) -> Path:
    if args.output:
        return resolve_path(args.output)

    retrieval_slug = slugify(retrieval_path.stem)
    filename = f"rag__{slugify(args.provider)}__{slugify(args.model)}__{retrieval_slug}__{utc_timestamp_slug()}.csv"
    return resolve_path(args.runs_dir) / filename


def parse_retrieved_passages(value) -> list[dict]:
    if value is None:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def select_passage_texts(retrieved_passages: list[dict], max_passages: int, max_passage_chars: int) -> list[str]:
    selected = []
    for passage_record in retrieved_passages[:max_passages]:
        passage = str(passage_record.get("passage") or "").strip()
        if not passage:
            continue
        selected.append(passage[:max_passage_chars])
    return selected


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def main() -> None:
    args = parse_args()

    retrieval_path = resolve_path(args.retrieval)
    if not retrieval_path.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}")

    output_path = build_output_path(args, retrieval_path)
    ensure_dir(output_path.parent)

    retrieval_df = pd.read_csv(retrieval_path)
    if args.limit is not None:
        retrieval_df = retrieval_df.head(args.limit).copy()

    existing_df = pd.DataFrame()
    completed_question_ids: set[str] = set()
    if output_path.exists() and not args.overwrite:
        existing_df = pd.read_csv(output_path)
        if "question_id" in existing_df.columns:
            completed_question_ids = {
                str(question_id)
                for question_id in existing_df["question_id"].dropna().astype(str).tolist()
            }

    pending_df = retrieval_df[
        ~retrieval_df["question_id"].astype(str).isin(completed_question_ids)
    ].copy()

    rag_config = RagLLMConfig(
        model_name=args.model,
        system_prompt=args.system_prompt,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        retry_max_output_tokens=args.retry_max_output_tokens,
    )
    answerer = create_rag_answerer(args.provider, rag_config)

    result_rows: list[dict[str, object]] = []
    for row in tqdm(
        pending_df.to_dict(orient="records"),
        total=len(pending_df),
        desc="Generating RAG answers",
    ):
        question = str(row.get("question") or "").strip()
        retrieved_passages = parse_retrieved_passages(row.get("retrieved_passages_json"))
        passage_texts = select_passage_texts(
            retrieved_passages,
            max_passages=args.max_passages,
            max_passage_chars=args.max_passage_chars,
        )

        result_row = {
            "question_id": row.get("question_id"),
            "question": question,
            "gold_primary": row.get("gold_primary"),
            "gold_aliases_json": row.get("gold_aliases_json"),
            "provider": args.provider,
            "model_name": args.model,
            "system_prompt": args.system_prompt,
            "retrieval_file": str(retrieval_path),
            "max_passages": args.max_passages,
            "max_passage_chars": args.max_passage_chars,
            "evidence_chunk_count": row.get("evidence_chunk_count"),
            "gold_supported_in_top_k": row.get("gold_supported_in_top_k"),
            "supported_gold_answer": safe_text(row.get("supported_gold_answer", "")),
            "supported_rank": row.get("supported_rank"),
            "evidence_passages_json": json.dumps(passage_texts, ensure_ascii=False),
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
            generation = answerer.generate_answer(question, passage_texts, row)
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
        combined_df = pd.concat([existing_df, pd.DataFrame(result_rows)], ignore_index=True)
        combined_df.to_csv(output_path, index=False)

    combined_df = pd.concat([existing_df, pd.DataFrame(result_rows)], ignore_index=True)
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "run_type": "rag",
        "provider": args.provider,
        "model_name": args.model,
        "retrieval_path": str(retrieval_path),
        "output_path": str(output_path),
        "requested_rows": int(len(retrieval_df)),
        "completed_rows": int(len(combined_df)),
        "resumed_rows": int(len(existing_df)),
        "new_rows": int(len(result_rows)),
        "failed_rows": int((combined_df["generation_error"].fillna("").str.strip() != "").sum())
        if "generation_error" in combined_df
        else 0,
        "empty_prediction_rows": int((combined_df["predicted_answer"].fillna("").str.strip() == "").sum())
        if "predicted_answer" in combined_df
        else 0,
        "max_passages": args.max_passages,
        "max_passage_chars": args.max_passage_chars,
    }
    write_json(summary, summary_path)

    print(f"Wrote RAG predictions to {output_path}")
    print(f"Wrote RAG summary to {summary_path}")


if __name__ == "__main__":
    main()
