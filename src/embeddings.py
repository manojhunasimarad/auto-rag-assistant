"""Pluggable embedding backends.

Two implementations behind one small interface (`fit`, `embed`):

- `SentenceTransformerEmbedder` — real dense semantic embeddings from a
  pretrained transformer (`all-MiniLM-L6-v2` by default: 384 dimensions,
  small enough to run on CPU, good enough retrieval quality for this size
  of corpus). This is what "semantic search" in the JD means, and what
  you'd actually demo. Requires the `sentence-transformers` package and,
  the first time it runs, a one-time model download.

- `TfidfEmbedder` — classic sparse bag-of-words vectors via scikit-learn's
  TF-IDF, densified. It is NOT semantic search (it matches on shared
  vocabulary, not meaning — "throttle position" and "accelerator angle"
  would not score as similar even though they mean roughly the same
  thing here), and the code says so at the point it's constructed. It
  exists purely so the rest of the pipeline (chunking -> embed -> index ->
  retrieve -> prompt -> answer) can be exercised end-to-end in a unit test
  or CI runner that has scikit-learn but no network access and no GPU/model
  download — which is exactly the situation this project's automated tests
  run under.

Both implementations satisfy the same interface on purpose, so
`rag_chain.py` and `scripts/build_index.py` never need to know which one
they're talking to.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


class SentenceTransformerEmbedder:
    """Real semantic embeddings. Requires `pip install sentence-transformers`."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "sentence-transformers is not installed. Run this project's "
                "`uv sync` on a machine with internet access, or set "
                "EMBEDDER_BACKEND=tfidf in .env to use the dependency-free "
                "fallback instead."
            ) from exc

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        # Pretrained model: nothing to fit. Kept as a no-op so this class
        # satisfies the same interface as TfidfEmbedder.
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        )

    def save(self, path: Path) -> None:
        # Nothing to persist beyond the model name, which build_index.py
        # already writes into index/meta.json — re-loading just re-downloads
        # (or reuses the local HuggingFace cache for) the same named model.
        return None


class TfidfEmbedder:
    """Sparse bag-of-words fallback. No network, no model download.

    NOTE: this is a *keyword-overlap* representation, not a semantic one.
    It's here so tests and CI can exercise the full pipeline without a
    model download or network access -- not because it's a good retrieval
    model. See the module docstring.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=4096)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit(corpus) must be called before embed()")
        return self._vectorizer.transform(list(texts)).toarray().astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self._vectorizer, f)

    @classmethod
    def load(cls, path: Path) -> "TfidfEmbedder":
        embedder = cls()
        with open(Path(path) / "tfidf_vectorizer.pkl", "rb") as f:
            embedder._vectorizer = pickle.load(f)
        embedder._fitted = True
        return embedder


def get_embedder(backend: str, model_name: str = "") -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown EMBEDDER_BACKEND: {backend!r} (expected 'sentence-transformers' or 'tfidf')")


def load_embedder(backend: str, model_name: str, index_dir: Path) -> Embedder:
    """Reconstruct a previously-fit embedder from an on-disk index."""
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "tfidf":
        return TfidfEmbedder.load(index_dir)
    raise ValueError(f"Unknown EMBEDDER_BACKEND: {backend!r}")
