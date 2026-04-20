from __future__ import annotations

from dataclasses import dataclass
import json
import random

import pandas as pd

from src.data.loader import extract_triviaqa_answer_metadata, flatten_sequence_dict


@dataclass
class PilotBenchmarkConfig:
    sample_size: int = 100
    random_seed: int = 42
    top_search_results: int = 3
    top_entity_pages: int = 2


def _json_text(value: list[str]) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_triviaqa_pilot_frame(dataset_split, config: PilotBenchmarkConfig) -> pd.DataFrame:
    """Create a small dataframe for early inspection and pilot experiments."""
    rows = []
    dataset_length = len(dataset_split)
    if dataset_length == 0:
        return pd.DataFrame(rows)

    rng = random.Random(config.random_seed)
    candidate_count = min(dataset_length, max(config.sample_size * 3, config.sample_size))
    candidate_indices = sorted(rng.sample(range(dataset_length), candidate_count))
    candidate_rows = dataset_split.select(candidate_indices)

    for row in candidate_rows:
        question = str(row.get("question") or "").strip()
        answer_metadata = extract_triviaqa_answer_metadata(row.get("answer"))
        answer_aliases = answer_metadata["aliases"]

        if not question or not answer_aliases:
            continue

        search_results = flatten_sequence_dict(
            row.get("search_results"),
            limit=config.top_search_results,
        )
        entity_pages = flatten_sequence_dict(
            row.get("entity_pages"),
            limit=config.top_entity_pages,
        )

        search_titles = [str(record.get("title") or "").strip() for record in search_results if record.get("title")]
        search_urls = [str(record.get("url") or "").strip() for record in search_results if record.get("url")]
        search_contexts = [
            str(record.get("search_context") or "").strip()
            for record in search_results
            if record.get("search_context")
        ]
        entity_titles = [str(record.get("title") or "").strip() for record in entity_pages if record.get("title")]
        entity_contexts = [
            str(record.get("wiki_context") or "").strip()
            for record in entity_pages
            if record.get("wiki_context")
        ]

        rows.append(
            {
                "question_id": row.get("question_id"),
                "question_source": str(row.get("question_source") or "").strip(),
                "question": question,
                "gold_primary": answer_metadata["primary_value"] or answer_aliases[0],
                "gold_primary_normalized": answer_metadata["normalized_value"],
                "gold_alias_count": len(answer_aliases),
                "gold_aliases_json": _json_text(answer_aliases),
                "gold_normalized_aliases_json": _json_text(answer_metadata["normalized_aliases"]),
                "answer_type": answer_metadata["answer_type"],
                "matched_wiki_entity_name": answer_metadata["matched_wiki_entity_name"],
                "normalized_matched_wiki_entity_name": answer_metadata["normalized_matched_wiki_entity_name"],
                "search_result_count": len(search_results),
                "search_titles_json": _json_text(search_titles),
                "search_urls_json": _json_text(search_urls),
                "search_contexts_json": _json_text(search_contexts),
                "entity_page_count": len(entity_pages),
                "entity_titles_json": _json_text(entity_titles),
                "entity_contexts_json": _json_text(entity_contexts),
            }
        )

        if len(rows) >= config.sample_size:
            break

    return pd.DataFrame(rows)
