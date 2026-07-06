"""
Embedding creation + vector store.

Two embedding backends are supported:

  - "tfidf" (default): scikit-learn TF-IDF vectors. Works fully offline,
    no model download required. Good baseline for keyword-heavy queries.

  - "sentence-transformers": a real neural embedding model
    (all-MiniLM-L6-v2) that captures semantic meaning, not just keyword
    overlap. Needs internet access the first time to download the model
    from Hugging Face (cached locally after that). Use this for better
    retrieval quality when you have internet access.

Both backends expose the same VectorStore interface, and retrieval is
done with cosine similarity over a simple in-memory numpy index -- no
external vector DB service needed for small/medium document collections.
"""
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


class _TfidfBackend:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return self._normalize(matrix.toarray().astype(np.float32))

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF-IDF vectorizer has not been fit yet. Call build() first.")
        matrix = self.vectorizer.transform(texts)
        return self._normalize(matrix.toarray().astype(np.float32))

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class _SentenceTransformerBackend:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def _make_backend(backend_name: str, model_name: Optional[str] = None):
    if backend_name == "tfidf":
        return _TfidfBackend()
    elif backend_name == "sentence-transformers":
        return _SentenceTransformerBackend(model_name or "all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding backend: {backend_name}")


class VectorStore:
    def __init__(self, backend: str = "tfidf", model_name: Optional[str] = None):
        self.backend_name = backend
        self.model_name = model_name
        self.backend = _make_backend(backend, model_name)
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def build(self, chunks: List[Dict]) -> None:
        """Fit the embedder on this corpus and index the given chunks."""
        self.chunks = list(chunks)
        texts = [c["text"] for c in chunks]
        self.embeddings = self.backend.fit_transform(texts)

    def add(self, chunks: List[Dict]) -> None:
        """
        Add more chunks to an existing index.
        Note: for the TF-IDF backend this re-fits the vectorizer on the
        full (old + new) corpus, since TF-IDF vocabularies are corpus-dependent.
        """
        self.chunks.extend(chunks)
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.backend.fit_transform(texts)

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        """Return top_k chunks most similar to the query, with similarity scores."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_vec = self.backend.transform([query])[0]
        scores = self.embeddings @ query_vec  # cosine similarity (vectors are normalized)
        top_k = min(top_k, len(self.chunks))
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            record = dict(self.chunks[idx])
            record["score"] = float(scores[idx])
            results.append(record)
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "backend_name": self.backend_name,
                "model_name": self.model_name,
                "backend": self.backend,
                "chunks": self.chunks,
                "embeddings": self.embeddings,
            }, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls.__new__(cls)
        store.backend_name = data["backend_name"]
        store.model_name = data["model_name"]
        store.backend = data["backend"]
        store.chunks = data["chunks"]
        store.embeddings = data["embeddings"]
        return store
