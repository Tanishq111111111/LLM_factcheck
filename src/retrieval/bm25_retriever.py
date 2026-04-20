from __future__ import annotations

from dataclasses import dataclass
import re

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Retriever:
    corpus: list[str]

    def __post_init__(self) -> None:
        tokenized_corpus = [tokenize_for_bm25(document) for document in self.corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        tokenized_query = tokenize_for_bm25(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.corpus, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]
