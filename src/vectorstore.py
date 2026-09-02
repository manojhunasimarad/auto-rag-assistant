"""A small, dependency-light vector store.

Why hand-roll this instead of using Chroma/FAISS/Pinecone
-------------------------------------------------------------
For a corpus of a few hundred to a few thousand chunks (this project's
whole point is a handful of engineering documents, not a web-scale
corpus), brute-force cosine similarity over a NumPy matrix *is* an exact
nearest-neighbour search, it has zero extra dependencies, it's fast enough
to answer a query in milliseconds, and — importantly for this repo — it's
fully unit-testable in any Python environment with just numpy installed,
no vector-database service, no network, no model download.

This is a real architectural trade-off and it's worth being able to state
the other side of it in an interview: brute-force search is O(n) per
query in the number of stored vectors. Once a corpus gets into the
hundreds of thousands to millions of chunks, you want an approximate
nearest-neighbour (ANN) index — this is exactly what Chroma, FAISS, or a
managed vector DB give you (HNSW or IVF indexes trading a small amount of
recall for sub-linear query time). The `VectorStore` interface below is
deliberately narrow (`add`, `query`, `save`, `load`) so it could be
swapped for a Chroma-backed implementation without changing any calling
code in rag_chain.py or agent.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QueryResult:
    id: str
    score: float
    document: str
    metadata: dict


class VectorStore:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None  # shape (n, d), L2-normalized
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not (len(ids) == len(vectors) == len(documents) == len(metadatas)):
            raise ValueError("ids, vectors, documents, metadatas must be same length")

        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero for a degenerate zero vector
        vectors = vectors / norms

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)

    def query(self, query_vector: np.ndarray, k: int = 4) -> list[QueryResult]:
        if self._vectors is None or len(self._ids) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        # Cosine similarity == dot product, since both sides are unit vectors.
        scores = self._vectors @ q
        k = min(k, len(self._ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]  # sort just the top-k

        return [
            QueryResult(
                id=self._ids[i],
                score=float(scores[i]),
                document=self._documents[i],
                metadata=self._metadatas[i],
            )
            for i in top_idx
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(path / "vectors.npy", self._vectors)
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "ids": self._ids,
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        path = Path(path)
        store = cls()
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        store._ids = meta["ids"]
        store._documents = meta["documents"]
        store._metadatas = meta["metadatas"]
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            store._vectors = np.load(vectors_path)
        return store
