"""
sparse.py — TF-IDF 稀疏检索器
===============================

Ponytail: sklearn TfidfVectorizer, char_wb n-grams 2-4,
适合学术论文中的公式/技术术语/混合中英文。
"""
from __future__ import annotations


class SparseRetriever:
    """TF-IDF based sparse retriever. Builds index once, queries many times."""

    def __init__(self):
        self._vectorizer = None
        self._matrix = None
        self._chunk_ids: list[str] = []
        self._documents: list[str] = []

    def index(self, chunks: list[dict]):
        """Build TF-IDF matrix from chunk contents.

        Args:
            chunks: List of {chunk_id, content, ...} dicts (rag_chunks.json schema).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._documents = [c.get("content", "") for c in chunks]
        # ponytail: character n-grams handle formula/math tokens better
        self._vectorizer = TfidfVectorizer(
            max_features=10000, analyzer="char_wb", ngram_range=(2, 4),
        )
        self._matrix = self._vectorizer.fit_transform(self._documents)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return list of {chunk_id, score}."""
        import numpy as np

        if self._matrix is None:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ q_vec.T).toarray().flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk_id": self._chunk_ids[i], "score": float(scores[i])}
            for i in top_idx if scores[i] > 0
        ]
