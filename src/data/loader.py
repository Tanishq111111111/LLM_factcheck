from __future__ import annotations

from typing import Any


def load_hf_dataset(dataset_name: str, config_name: str | None = None):
    """Load a Hugging Face dataset.

    Import is local so the rest of the repo can be used without the
    dependency until dataset work actually starts.
    """
    from datasets import load_dataset

    return load_dataset(dataset_name, config_name)


def load_dataset_split(dataset_name: str, split: str, config_name: str | None = None):
    from datasets import load_dataset

    return load_dataset(dataset_name, config_name, split=split)


def extract_triviaqa_aliases(answer_field: Any) -> list[str]:
    """Collect likely valid answer aliases from a TriviaQA answer object."""
    if not isinstance(answer_field, dict):
        return []

    ordered_candidates = [
        answer_field.get("value"),
        answer_field.get("normalized_value"),
        *(answer_field.get("aliases") or []),
    ]

    aliases: list[str] = []
    seen: set[str] = set()
    for candidate in ordered_candidates:
        if not candidate:
            continue
        text = str(candidate).strip()
        if not text or text in seen:
            continue
        aliases.append(text)
        seen.add(text)
    return aliases


def extract_triviaqa_answer_metadata(answer_field: Any) -> dict[str, Any]:
    """Flatten the TriviaQA answer object into a smaller metadata dict."""
    if not isinstance(answer_field, dict):
        return {
            "primary_value": "",
            "normalized_value": "",
            "aliases": [],
            "normalized_aliases": [],
            "matched_wiki_entity_name": "",
            "normalized_matched_wiki_entity_name": "",
            "answer_type": "",
        }

    return {
        "primary_value": str(answer_field.get("value") or "").strip(),
        "normalized_value": str(answer_field.get("normalized_value") or "").strip(),
        "aliases": extract_triviaqa_aliases(answer_field),
        "normalized_aliases": [
            str(alias).strip()
            for alias in (answer_field.get("normalized_aliases") or [])
            if str(alias).strip()
        ],
        "matched_wiki_entity_name": str(answer_field.get("matched_wiki_entity_name") or "").strip(),
        "normalized_matched_wiki_entity_name": str(
            answer_field.get("normalized_matched_wiki_entity_name") or ""
        ).strip(),
        "answer_type": str(answer_field.get("type") or "").strip(),
    }


def flatten_sequence_dict(sequence_field: Any, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a Hugging Face sequence-of-fields dict into a list of records."""
    if not isinstance(sequence_field, dict):
        return []

    list_lengths = [len(value) for value in sequence_field.values() if isinstance(value, list)]
    if not list_lengths:
        return []

    record_count = max(list_lengths)
    if limit is not None:
        record_count = min(record_count, limit)

    records: list[dict[str, Any]] = []
    for index in range(record_count):
        record: dict[str, Any] = {}
        for key, value in sequence_field.items():
            if isinstance(value, list):
                record[key] = value[index] if index < len(value) else None
            else:
                record[key] = value

        if any(item not in ("", None, []) for item in record.values()):
            records.append(record)

    return records
