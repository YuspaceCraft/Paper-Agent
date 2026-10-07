"""
sparse.py — TF-IDF 稀疏检索器
===============================

Ponytail: sklearn TfidfVectorizer, char_wb n-grams 2-4,
适合学术论文中的公式/技术术语/混合中英文。
"""
from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path


_CACHE_VERSION = 1
_INDEX_SIGNATURE = "tfidf-char_wb-2-4-max10000"


class SparseRetriever:
    """TF-IDF based sparse retriever. Builds index once, queries many times."""

    def __init__(self):
        self._vectorizer = None
        self._matrix = None
        self._chunk_ids: list[str] = []
        self._documents: list[str] = []

    def index(
        self,
        chunks: list[dict],
        *,
        cache_path: str | Path | None = None,
        cache_key: str = "",
    ):
        """Build TF-IDF matrix from chunk contents.

        Args:
            chunks: List of {chunk_id, content, ...} dicts (rag_chunks.json schema).
            cache_path: Optional local pickle cache for the fitted matrix.
            cache_key: Fingerprint of the source chunks. Required for cache reuse.
        """
        cache_file = Path(cache_path) if cache_path else None
        if cache_file and cache_key and self._load_cache(cache_file, cache_key):
            return

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._documents = [c.get("content", "") for c in chunks]
        if not self._documents:
            self._vectorizer = None
            self._matrix = None
            return

        # ponytail: character n-grams handle formula/math tokens better
        self._vectorizer = TfidfVectorizer(
            max_features=10000, analyzer="char_wb", ngram_range=(2, 4),
        )
        self._matrix = self._vectorizer.fit_transform(self._documents)
        if cache_file and cache_key:
            try:
                self._save_cache(cache_file, cache_key)
            except Exception as exc:  # noqa: BLE001 - cache is optional
                print(
                    f"[SPARSE] cache write skipped: "
                    f"{type(exc).__name__}: {exc}"
                )

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

    def _load_cache(self, path: Path, cache_key: str) -> bool:
        """Load a compatible fitted index. Corrupt or stale caches are ignored."""
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                return False
            if payload.get("cache_version") != _CACHE_VERSION:
                return False
            if payload.get("index_signature") != _INDEX_SIGNATURE:
                return False
            if payload.get("cache_key") != cache_key:
                return False

            vectorizer = payload["vectorizer"]
            matrix = payload["matrix"]
            chunk_ids = list(payload["chunk_ids"])
            documents = list(payload["documents"])
            if matrix.shape[0] != len(chunk_ids) or len(chunk_ids) != len(documents):
                return False

            self._vectorizer = vectorizer
            self._matrix = matrix
            self._chunk_ids = chunk_ids
            self._documents = documents
            return True
        except (OSError, EOFError, pickle.PickleError, AttributeError, KeyError,
                TypeError, ValueError, ImportError):
            return False

    def _save_cache(self, path: Path, cache_key: str) -> None:
        """Atomically persist the fitted index so process restarts stay warm."""
        payload = {
            "cache_version": _CACHE_VERSION,
            "index_signature": _INDEX_SIGNATURE,
            "cache_key": cache_key,
            "vectorizer": self._vectorizer,
            "matrix": self._matrix,
            "chunk_ids": self._chunk_ids,
            "documents": self._documents,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with tmp_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
