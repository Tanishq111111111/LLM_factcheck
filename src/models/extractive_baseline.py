from collections import Counter


def lexical_overlap_score(question: str, passage: str) -> int:
    question_tokens = question.lower().split()
    passage_tokens = passage.lower().split()
    counts = Counter(passage_tokens)
    return sum(counts[token] for token in question_tokens)


def pick_best_passage(question: str, passages: list[str]) -> str:
    if not passages:
        return ""
    return max(passages, key=lambda passage: lexical_overlap_score(question, passage))
