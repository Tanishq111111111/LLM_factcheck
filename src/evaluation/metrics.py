from __future__ import annotations

from collections import Counter

from src.data.normalize_answers import normalize_answer_list, normalize_text


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    return prediction in set(gold_answers)


def normalized_exact_match(prediction: str, gold_answers: list[str]) -> bool:
    normalized_answers = set(normalize_answer_list(gold_answers))
    return normalize_text(prediction) in normalized_answers


def token_f1(prediction: str, gold_answer: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold_answer).split()

    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(prediction_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(token_f1(prediction, answer) for answer in gold_answers)
