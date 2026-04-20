from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class BM25Retriever:
    corpus: list[str]

    def __post_init__(self) -> None:
        tokenized_corpus = [document.split() for document in self.corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        tokenized_query = query.split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.corpus, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]
