import re
import string


ARTICLES_PATTERN = re.compile(r"\b(a|an|the)\b")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Normalize answer text for lightweight matching."""
    if text is None:
        return ""

    normalized = text.lower().strip()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = ARTICLES_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def normalize_answer_list(answers) -> list[str]:
    if answers is None:
        return []
    return [normalize_text(answer) for answer in answers if answer]


def contains_normalized_match(prediction: str, answers) -> bool:
    normalized_prediction = normalize_text(prediction)
    return normalized_prediction in set(normalize_answer_list(answers))
