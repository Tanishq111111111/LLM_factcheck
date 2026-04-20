from __future__ import annotations

import json
import re
from typing import Any

from src.data.normalize_answers import normalize_text
from src.retrieval.chunking import chunk_text


WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


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
    return [str(parsed)] if str(parsed).strip() else []


def clean_evidence_text(text: str) -> str:
    cleaned = CONTROL_PATTERN.sub(" ", str(text))
    cleaned = cleaned.replace("\ufffd", " ")
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def build_evidence_chunks(
    row: dict,
    chunk_size: int,
    overlap: int,
    include_search_contexts: bool = True,
    include_entity_contexts: bool = True,
    max_chunks: int | None = None,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []

    evidence_sources: list[tuple[str, list[str]]] = []
    if include_search_contexts:
        evidence_sources.append(("search", parse_json_list(row.get("search_contexts_json"))))
    if include_entity_contexts:
        evidence_sources.append(("entity", parse_json_list(row.get("entity_contexts_json"))))

    for source_name, contexts in evidence_sources:
        for context_index, context in enumerate(contexts):
            cleaned_context = clean_evidence_text(context)
            for chunk_index, chunk in enumerate(chunk_text(cleaned_context, chunk_size, overlap)):
                if not chunk:
                    continue
                chunks.append(
                    {
                        "source": source_name,
                        "context_index": context_index,
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                )
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks

    return chunks


def find_supported_gold_answer(passage: str, gold_answers: list[str]) -> str:
    normalized_passage = normalize_text(passage)
    for answer in gold_answers:
        normalized_answer = normalize_text(answer)
        if normalized_answer and normalized_answer in normalized_passage:
            return answer
    return ""
